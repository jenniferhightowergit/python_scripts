'''
append_training_orders pipeline

Copies work orders from the live tables into the training tables,
so they survive the 3-hour retention on orders / cleaned_text,
then rebuilds the wide training dataset (one column per category).

    orders        --> training_orders
    cleaned_text  --> training_cleaned_text
    training_order_category (labels, long)  --> training_dataset (wide)

1. find orders (wr_id + fa_id) with cleaned text, not in training_orders yet
2. write them to training_orders (training_order_id comes from its sequence)
3. write their cleaned text to training_cleaned_text
4. rebuild training_dataset: text + one label column per category
   (1 / 0 / null = not labeled yet)

No retention: training tables only grow.
The labeling pipeline fills in training_order_category.

NOTE: this must run more often than the orders retention window (3h),
or orders will be deleted before they get copied.

run from src/:
>>> python main.py pipelines run -n append-training-orders
'''

import logging

# Max new orders appended per run (None = no cap)
DEFAULT_N = 100


# --------------------------------------------------------------------------
# 1-2. orders --> training_orders
# --------------------------------------------------------------------------

# Rows from orders that are ready to copy:
#   - have cleaned text already (anything not cleaned yet gets picked up next run)
#   - (wr_id, fa_id) not in training_orders yet
#   - one row per (wr_id, fa_id), latest update wins, so the
#     uq_order_business_key constraint can't be violated
# comments / instructions are NOT NULL in training_orders, and Oracle turns
# blank text into NULL, so blanks are stored as '' instead of being dropped.
NEW_ORDERS_SQL = '''
select
    o.wr_id,
    o.fa_id,
    o.wr_status_upd_dttm,
    coalesce(o.comments, '')     as comments,
    coalesce(o.instructions, '') as instructions
from orders o
where o.wr_id is not null
  and o.fa_id is not null
  and exists (
      select 1 from cleaned_text ct where ct.order_id = o.order_id
  )
  and not exists (
      select 1 from training_orders t
      where t.wr_id = o.wr_id
        and t.fa_id = o.fa_id
  )
qualify row_number() over (
    partition by o.wr_id, o.fa_id
    order by o.wr_status_upd_dttm desc, o.order_id desc
) = 1
order by o.wr_status_upd_dttm
{limit}
'''


def append_training_orders(tables, n: int | None = None) -> int:
    limit = f'limit {int(n)}' if n else ''
    sql = NEW_ORDERS_SQL.format(limit=limit)

    n_new = count_rows(tables, sql)
    if n_new == 0:
        logging.info('no new orders to append to "training_orders"')
        return 0

    tables.execute(f'''
    insert into training_orders (wr_id, fa_id, wr_status_upd_dttm, comments, instructions)
    {sql}
    ''')
    logging.info(f'inserted: {n_new:,} records to "training_orders"')
    return n_new


# --------------------------------------------------------------------------
# 3. cleaned_text --> training_cleaned_text
# --------------------------------------------------------------------------

# training_cleaned_text is keyed on training_order_id, so each training order
# is matched back to orders on (wr_id, fa_id) to find its cleaned text.
# TODO: assumes cleaned_text columns are named cleaned_comments /
#       cleaned_instructions (prediction reads CLEANED_COMMENTS).
#       Confirm with: describe cleaned_text
NEW_TEXT_SQL = '''
select
    t.training_order_id,
    coalesce(ct.cleaned_comments, '')     as cleaned_comments,
    coalesce(ct.cleaned_instructions, '') as cleaned_instructions
from training_orders t
join orders o
  on o.wr_id = t.wr_id
 and o.fa_id = t.fa_id
join cleaned_text ct
  on ct.order_id = o.order_id
where not exists (
    select 1 from training_cleaned_text tc
    where tc.training_order_id = t.training_order_id
)
qualify row_number() over (
    partition by t.training_order_id
    order by o.order_id desc
) = 1
'''


def append_training_text(tables) -> int:
    n_new = count_rows(tables, NEW_TEXT_SQL)
    if n_new == 0:
        logging.info('no new text to append to "training_cleaned_text"')
        return 0

    tables.execute(f'''
    insert into training_cleaned_text (training_order_id, cleaned_comments, cleaned_instructions)
    {NEW_TEXT_SQL}
    ''')
    logging.info(f'inserted: {n_new:,} records to "training_cleaned_text"')
    return n_new


# --------------------------------------------------------------------------
# 4. wide training dataset: one column per category
# --------------------------------------------------------------------------

# Every training order x every category, with the label where one exists.
# The cross join means a brand-new category shows up as a column of nulls
# before anyone has labeled it.
LABEL_GRID_SQL = '''
select
    t.training_order_id,
    c.code,
    l.manual_label
from training_orders t
cross join category c
left join training_order_category l
       on l.training_order_id = t.training_order_id
      and l.category_id = c.category_id
'''


def build_training_dataset(tables) -> int:
    '''
    Rebuilds training_dataset from scratch.

    Tables, not views: DuckDB can't save a view whose pivot columns
    come from the data, since the column list changes as categories are added.
    '''
    if count_rows(tables, LABEL_GRID_SQL) == 0:
        logging.info('no training orders or no categories yet: skipped "training_dataset"')
        return 0

    tables.execute(f'''
    create or replace table training_labels_wide as
    with grid as ({LABEL_GRID_SQL})
    pivot grid
    on code
    using first(manual_label)
    group by training_order_id
    ''')

    tables.execute('''
    create or replace table training_dataset as
    select
        t.training_order_id,
        t.wr_id,
        t.fa_id,
        tc.cleaned_comments,
        tc.cleaned_instructions,
        w.* exclude (training_order_id)
    from training_orders t
    join training_cleaned_text tc using (training_order_id)
    join training_labels_wide w using (training_order_id)
    order by t.training_order_id
    ''')

    n = count_rows(tables, 'select * from training_dataset')
    logging.info(f'rebuilt "training_dataset": {n:,} records')
    return n


# --------------------------------------------------------------------------
# helpers + entrypoint
# --------------------------------------------------------------------------

def count_rows(tables, sql: str) -> int:
    df = tables.select(f'select count(*) as n from ({sql})')
    return int(df.iloc[0, 0])


def run(n: int | None = DEFAULT_N):
    '''
    n: cap on how many new orders to append per run.
       None means append everything new.
    '''
    # imported here (not at the top) so this file can be loaded in a
    # test notebook without amslib / settings installed
    # TODO: confirm import path after the "Fixed Circular Imports" commit
    from database import init_tables
    from settings import get_settings

    logging.info('started: append_training_orders_pipeline')

    settings = get_settings()
    tables = init_tables(settings)

    append_training_orders(tables, n)
    append_training_text(tables)
    build_training_dataset(tables)

    logging.info('finished: append_training_orders_pipeline')

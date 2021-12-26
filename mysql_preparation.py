import mysql.connector as mysql

def build_connection(password,
                     host='localhost',
                     user='root',
                     database=''):
    db = mysql.connect(host=host,
                       user=user,
                       password=password,
                       database=database,
                       auth_plugin='mysql_native_password')
    return db

def return_cases(password,
                 exc_message,
                 host='localhost',
                 user='root',
                 database=''):
    db = build_connection(host=host,
                          user=user,
                          password=password,
                          database=database)
    
    try:
        if db.is_connected():
            cursor = db.cursor()
            cursor.execute(exc_message)
            cases = cursor.fetchall()
            cases = [d[0] for d in cases]
    except mysql.Error as e:
        print('connection error {}'.format(e))
        
    return cases,cursor,db

def insert_data(cursor,db,table_name):
    cursor.execute("load data infile '%s' into table amazon_train fields terminated by ',' lines terminated by '\n' ignore 1 rows" % table_name)
    db.commit()

if __name__ == '__main__':
    databases,cursor,_ = return_cases('111',
                                    exc_message="show databases")
    if 'kaggle' not in databases:
        cursor.execute('create database kaggle')
        
    tables,cursor,db = return_cases('111',
                                    database='kaggle',
                                    exc_message='show tables')
    if 'amazon_train' not in tables:
        table_description = (
                         "CREATE TABLE `amazon_train` ( "
                         "`action` tinyint unsigned not null, "
                         "`resource` mediumint unsigned not null, "
                         "`mgr_id` mediumint unsigned not null, "
                         "`role_rollup_1` mediumint unsigned not null, "
                         "`role_rollup_2` mediumint unsigned not null, "
                         "`role_deptname` mediumint unsigned not null, "
                         "`role_title` mediumint unsigned not null, "
                         "`role_family_desc` mediumint unsigned not null, "
                         "`role_family` mediumint unsigned not null, "
                         "`role_code` mediumint unsigned not null, "
                         "INDEX (`resource`) "
                         ") ENGINE=Innodb"
                                )
        cursor.execute(table_description)
    
    insert_data(cursor,db,'/python/mysql/train.csv')
    cursor.execute("ALTER TABLE amazon_train ADD COLUMN `id` INT AUTO_INCREMENT PRIMARY KEY")
    

    #%%
cursor.execute("alter table amazon_train add column `role_rollup` bigint unsigned not null")

cursor.execute('select role_rollup_1 from amazon_train')
role_rollup_1 = cursor.fetchall()
cursor.execute('select role_rollup_2 from amazon_train')
role_rollup_2 = cursor.fetchall()
role_rollup = [r1[0]+r2[0]*1000000 for r1,r2 in zip(role_rollup_1,role_rollup_2)]
    
for i,v in enumerate(role_rollup):
    if i % 5000 == 0:
        print(i)
    sql = "update amazon_train set role_rollup=%d where id=%d" % (v,i+1)
    cursor.execute(sql)

#cursor.execute('select * from tmp')
#z = cursor.fetchall()
#cursor.execute("insert into amazon_train (%s) select role_rollup from tmp"%'role_rollup')

#%%
import pandas as pd
import numpy as np
#train = pd.read_csv(r'train.csv')
#cursor.execute('drop table amazon_train')

cursor.execute('select * from amazon_train')
z = cursor.fetchall()
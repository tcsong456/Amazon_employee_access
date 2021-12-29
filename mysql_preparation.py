import mysql.connector as mysql
import pandas as pd
from helper.utils import logger

def raw_prep():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    if 'id' in test.columns:
        test.drop('id',axis=1,inplace=True)
    
    df_all = pd.concat([train,test]).reset_index(drop=True).fillna(0)
    
    test.to_csv('data/test.csv',index=False)
    df_all.to_csv('data/df_all.csv',index=False)

def build_connection(password,
                     host='localhost',
                     user='root',
                     database=''):
    db = mysql.connect(host=host,
                       user=user,
                       password=password,
                       database=database,
                       allow_local_infile=True,
                       auth_plugin='mysql_native_password')
    return db

def return_cases(password,
                 exc_message,
                 host='mysqldb',
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

def insert_data(cursor,db,table_name,table_file):
    cursor.execute("load data local infile '%s' into table %s fields terminated by ',' lines terminated by '\n' ignore 1 rows" % (table_file,table_name))
    db.commit()

create_table = lambda ds,extra: "CREATE TABLE IF NOT EXISTS `%s` (\
                                %s \
                                `resource` mediumint unsigned not null,\
                                `mgr_id` mediumint unsigned not null,\
                                `role_rollup_1` mediumint unsigned not null,\
                                `role_rollup_2` mediumint unsigned not null,\
                                `role_deptname` mediumint unsigned not null,\
                                `role_title` mediumint unsigned not null,\
                                `role_family_desc` mediumint unsigned not null,\
                                `role_family` mediumint unsigned not null,\
                                `role_code` mediumint unsigned not null,\
                                INDEX (`resource`) ) ENGINE=Innodb" % (ds,extra)

if __name__ == '__main__':
    logger.info('runnign mysql preparation!!!')
    raw_prep()
    
    databases,cursor,_ = return_cases('111',
                                      exc_message="show databases")
    if 'kaggle' not in databases:
        cursor.execute('create database kaggle')
        
    tables,cursor,db = return_cases('111',
                                    database='kaggle',
                                    exc_message='show tables')
    cursor.execute("show variables like 'local_infile'")
    local_infile = cursor.fetchall()
    cond = local_infile[0][1]
    if cond == 'OFF':
        cursor.execute("set global local_infile=1")
    
    if 'amazon_train' not in tables:
        logger.info('creating table amazon_train in mysql')
        
        table_description = create_table('amazon_train',"`action` tinyint unsigned not null,")
        cursor.execute(table_description)
        insert_data(cursor,db,'amazon_train','./data/train.csv')
        cursor.execute("ALTER TABLE amazon_train ADD COLUMN `id` INT AUTO_INCREMENT PRIMARY KEY")
    
    if 'amazon_test' not in tables:
        logger.info('creating table amazon_test in mysql')
        
        table_description = create_table('amazon_test',"")
        cursor.execute(table_description)
        insert_data(cursor,db,'amazon_test','./data/test.csv')
        cursor.execute("ALTER TABLE amazon_test ADD COLUMN `id` INT AUTO_INCREMENT PRIMARY KEY")
    
    if 'amazon_all' not in tables:
        logger.info('creating table amazon_all in mysql')
        
        table_description = create_table('amazon_all',"`action` tinyint unsigned not null,")
        cursor.execute(table_description)
        insert_data(cursor,db,'amazon_all','./data/df_all.csv')
        
    

    #%%

import pandas as pd
import mysql.connector as msc
from mysql.connector import Error

import json

def create_connection():
    try:
        params = json.load(open('config.json', 'r'))

        connection = msc.connect(
            host= params['host'],
            port= params['port'],
            user= params['user'],
            password= params['password'],
            database= params['database']
        )

        return connection

    except Error as e:
        print(f'Error: {e}')
        return None

def create_table(connection, table_name):
    cursor = connection.cursor()
    
    query = f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            stars INT NOT NULL,
            text VARCHAR(2000) NOT NULL
        )
    '''
    try:
        cursor.execute(query)

        print('Tabela criada com Sucesso!')
    except Error as e:
        print(f'Error: {e}')
    finally:
        cursor.close()


def select_values(connection, query):

    cursor = connection.cursor()

    cursor.execute(query)
    result = cursor.fetchall()


    return result


def insert_values(connection, df, table_name):
    cursor = connection.cursor()

    insert_query = f'''INSERT INTO {table_name} (stars, text) VALUES (%s, %s)'''

    data = list(df.itertuples(index= False, name= None))

    try:
        cursor.executemany(insert_query, data)

    except Error as e:
        print(f'Insert Error: {e}')

    finally:
        cursor.close()    


if __name__ == '__main__':
    connection = create_connection()

    table_name = 'YELP_REVIEW'
    json_path = 'dataset/yelp_academic_dataset_review.json'

    i = 0

    if connection.is_connected():   
        print('Conectado com sucesso!')

        create_table(connection, table_name)

        for chunk in pd.read_json(json_path, lines= True, chunksize= 10000):
            chunk = chunk[['stars', 'text']]
            chunk = chunk.loc[chunk['text'].str.len() < 2000]

            insert_values(connection, chunk, table_name)

            connection.commit()
            i += 1

            if i % 100 == 0:
                print(f"{i} chunks inserido com sucesso!")
        
        connection.close()







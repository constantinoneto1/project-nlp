import pandas as pd
import mysql.connector as msc
from mysql.connector import Error
import json

class MySQLCRUD:
    def __init__(self, config_path):
        self.connection = self.create_connection(config_path)
    
    def create_connection(self, config_path):
        try:
            params = json.load(open(config_path, 'r'))
            
            connection = msc.connect(
                host=params['host'],
                port=params['port'],
                user=params['user'],
                password=params['password'],
                database=params['database']
            )
            if connection.is_connected():
                print('Connected successfully!')
            return connection

        except Error as e:
            print(f'Error: {e}')
            return None

    def create_table(self, table_name):
        cursor = self.connection.cursor()
        
        query = f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                stars INT NOT NULL,
                text VARCHAR(2000) NOT NULL
            )
        '''
        try:
            cursor.execute(query)
            print('Table created successfully!')
        except Error as e:
            print(f'Error: {e}')
        finally:
            cursor.close()

    def insert_values(self, df, table_name):
        cursor = self.connection.cursor()

        insert_query = f'''INSERT INTO {table_name} (stars, text) VALUES (%s, %s)'''

        data = list(df.itertuples(index=False, name=None))

        try:
            cursor.executemany(insert_query, data)
            self.connection.commit()
            print('Data inserted successfully!')
        except Error as e:
            print(f'Insert Error: {e}')
        finally:
            cursor.close()

    def select_values(self, query):
        cursor = self.connection.cursor()

        cursor.execute(query)
        result = cursor.fetchall()

        cursor.close()
        return result

    def close_connection(self):
        if self.connection.is_connected():
            self.connection.close()
            print('Connection closed successfully!')

if __name__ == '__main__':
    config_path = 'D:/Documentos/Estudos/Projeto-NLP/config.json'
    table_name = 'YELP_REVIEW'
    json_path = 'dataset/yelp_academic_dataset_review.json'

    crud = MySQLCRUD(config_path)
    i = 0

    crud.create_table(table_name)

    for chunk in pd.read_json(json_path, lines=True, chunksize=10000):
        chunk = chunk[['stars', 'text']]
        chunk = chunk.loc[chunk['text'].str.len() < 2000]

        crud.insert_values(chunk, table_name)

        if i % 100 == 0:
            print(f"{i} chunks inserted successfully!")
        i += 1

    crud.close_connection()

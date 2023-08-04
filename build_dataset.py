import argparse
import requests
import numpy as np
from typing import List, Tuple, Dict, Any

def get_data(offset:int=0, cat:str='MLA')->dict:
    '''
    Fetch data from an specific category and offset using a query over the MELI api .
    Input:
        offset: Int to define the start point to fetch data
        cat: String that defines the desired category
    
    Output:
        Return a dict wich contains products data
    '''
    url = 'https://api.mercadolibre.com/sites/{cat}/search?q=tv%204k&offset={offset}'.format(cat=cat, offset=offset)
    r = requests.get(url, allow_redirects=True, stream=False)
    return r.json()

def get_all_items(cat:str='MLA')-> List[Tuple]:
    '''
    Try to fetch all possible data from an specific query over one categoty at the MELI api.
    Input:
        cat: String that defines the desired category
    
    Output:
        Return a list of tuples, (ITEM_ID, TITLE, PRICE, DOMAIN_ID, BRAND)
    '''
    ITEM_ID=[]
    TITLE=[]
    PRICE=[]
    DOMAIN_ID=[]
    BRAND=[]
    # MELI api limits offset to 1000, higher values it returns 403 : the requested offset is higher than the allowed for public users. Maximum allowed is 1000
    for offset in range(0,1001,50):
        data = get_data(offset)
        for i in range(len(data['results'])):
            try:
                if data['results'][i]['condition']=='new':
                    ITEM_ID.append(data['results'][i]['id'])
                    TITLE.append(data['results'][i]['title'])
                    PRICE.append(data['results'][i]['price'])
                    DOMAIN_ID.append(data['results'][i]['domain_id'])
                    BRAND.append([x['value_name'] for x in data['results'][i]['attributes'] if x['id']=='BRAND'][0])
            except IndexError as e:
                print(e.args, offset, i)
                pass
    return list(zip(ITEM_ID,TITLE,PRICE,DOMAIN_ID,BRAND))

def save_csv(rows: List[Tuple]) -> None:
    np.savetxt("dataset.csv", rows, delimiter =",", fmt ='% s')

def parse_args():
    '''
    Return parsed argument, first position is the category
    
    Output:
        Unique parameter, 
    '''
    msg = "Data Pipeline"

    # Initialize parser
    parser = argparse.ArgumentParser(description = msg)
    parser.add_argument('category', choices=['MLA', 'MLA', 'MLM']) 
    return parser.parse_args()

if __name__ == '__main__':
    save_csv(get_all_items(parse_args().category))
    #print(type(parse_args().category))
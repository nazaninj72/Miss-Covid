import pandas as pd 
from datetime import datetime
from dateutil.parser import parse
from bs4 import BeautifulSoup
import requests 
from utils import *
startdate=datetime.strptime('January 1 2020', '%B %d %Y')
class FactCheck:
    def __init__(self, fcname="politifact"):
        self.fcname = fcname
        if self.fcname=="politifact":
            self.searchurl = "https://www.politifact.com/search/?q="
        
        
    def crawl_politifact(self,topic,subtopic):
        
        if subtopic=="":
            r = requests.get(self.searchurl+topic+"+coronavirus+covid19")
        else:
            r = requests.get(self.searchurl+subtopic+"+coronavirus+covid19")
        
        soup = BeautifulSoup(r.text, 'html.parser')  
        factcheck_section = soup.find_all('section', attrs={'class':'o-platform o-platform--has-thin-border'})[2]
        results= factcheck_section.find_all('div', attrs={'class':'o-listease__item'})
        records = []  
        for result in results: 
            #either a person told it or a source of facebook post or twitter post, etc..
            item= result.find('div',attrs={'class':'m-result__content'})
            source=((item.find('div',attrs={'class':'c-textgroup__author'}).text).split("stated on ")[0]).replace("\n","")
            date = ((item.find('div',attrs={'class':'c-textgroup__author'}).text).split("stated on ")[1])
            date = date.split(" in")[0].replace(",","")
            d=datetime.strptime(date, '%B %d %Y')
            date=d.strftime('%Y-%m-%d')
            url = "https://www.politifact.com"+ result.find('div', attrs={'class':'c-textgroup__title'}).find('a')['href']
            statement,label=scrape_statement(url)

         
            
            if d>=startdate:
                records.append(( source,date,statement,url,topic,subtopic,label)) 
            
        return records

from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
import re
import pymysql.cursors
import os
from PIL import Image
from io import BytesIO
import sys
import configuration as conf

class flowerCollector:
    def __init__(self, link, destination, company):
        self.link = link
        self.destination = destination
        self.products = list()
        self.company = company
        if not os.path.exists(destination):
            os.makedirs(destination)

    def checkLink(self, link):
        """Checks if the link is valid
        """
        html = requests.get(self.link)
        try:
            html.raise_for_status()
        except Exception as e: 
            print(e)
            return -1
        return html
    
    def findProducts(self, attribute, attributeDetail, rootIncluded=False):
        """Collect product links from the provided link.

        Function targets the <div> tag with the user specified attribute and attribute name and adds to the 
        products list of the class

        Parameters
        ----------
        attribute : str
            The attribute the function targets 
        attributeDetail : str
            The attribute detail the function targets
        rootIncluded : boolean
            Select True if the root of the link is included 

        Returns
        -------
        None
        """
        html = self.checkLink(self.link)
        if html == -1: raise Exception("Link Problem")
        
        rootLink = self.link[:(self.link.find('com')+3)]

        bsObj = BeautifulSoup(html.content, "lxml")

        prod = set()
        for box in bsObj.findAll("div", {attribute: attributeDetail}):
            try:
                for a in box.findAll('a', href=True):
                    if rootIncluded:
                        prod.add(a['href'])
                    else:
                        prod.add(rootLink + a['href'])
            except Exception as e: 
                #print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
                pass

        self.products += list(prod)
    
    def dbConnect(self):
        """ Initialize connection
        """
        connection = pymysql.connect(host='localhost',
                    user='root',
                    password=conf.DB_PASS,
                    db='flower',
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor)
        return connection

    
    def scrapeProductsLadybug(self):
        """ Once a product is identified, we collect the following info:
        Product link, picture link, product name, price and save the images into 
        a folder and rest of the info into a DB
        """

        connection = self.dbConnect()

        try:
            with connection.cursor() as cursor:
                for prodLink in self.products:
                    html = requests.get(prodLink)
                    bsObj = BeautifulSoup(html.content, "lxml")

                    for content in bsObj.find("div", {"class":"col-md-12"}):
                        try:
                            imgLink = content.img.attrs['src']
                            prodName = content.h2.text
                            rawPrice = content.p.text
                            price = float(rawPrice[rawPrice.find('$')+1:])

                            r = requests.get(imgLink)
                            i = Image.open(BytesIO(r.content))
                            saveLoc = self.destination+'\\'+ imgLink.split('/')[-1]
                            i.save(saveLoc)

                            sql = "INSERT INTO `content` (`company`, `name`, `prodLink`, \
                                    `imgLink`, `imgLoc`, `price`) VALUES (%s, %s, %s, %s, %s, %s)"
                            cursor.execute(sql, (self.company, prodName, prodLink, imgLink, saveLoc, price))
                            connection.commit()
                        except Exception as e: 
                            #print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
                            pass
        finally:
            connection.close()
    
    def scrapeProductsMartin(self):
        """ Once a product is identified, we collect the following info:
        Product link, picture link, product name, price and save the images into 
        a folder and rest of the info into a DB
        """
        connection = self.dbConnect()

        try:
            with connection.cursor() as cursor:
                for prodLink in self.products:
                    html = requests.get(prodLink)
                    bsObj = BeautifulSoup(html.content, "lxml")

                    try:
                        for content in bsObj.find("div", {"class": "ccm-core-commerce-add-to-cart"}):
                            try:
                                imgLink = content.img.attrs['src']
                                prodName = content.h2.text
                                rawPrice = content.span.text
                                price = float(rawPrice[rawPrice.find('$')+1:])

                                saveLoc = self.destination+'\\'+ imgLink.split('/')[-1]

                                sql = "INSERT INTO `content` (`company`, `name`, `prodLink`, \
                                        `imgLink`, `imgLoc`, `price`) VALUES (%s, %s, %s, %s, %s, %s)"
                                cursor.execute(sql, (self.company, prodName, prodLink, imgLink, saveLoc, price))
                                connection.commit()
                                
                                r = requests.get(imgLink)
                                i = Image.open(BytesIO(r.content))
                                i.save(saveLoc)                           
                            except Exception as e: 
                                #print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
                                pass
                    except Exception as e:
                        print(e)
                        print(prodLink)
        finally:
            connection.close()         

if __name__ == '__main__':
    f = open('collect.txt', 'r')
    LINKS = [x.strip() for x in f.readlines()]

    scrapper = flowerCollector(link=None, destination='D:\\Data\\Flower\\martin', company='Martins Flowers')

    for link in LINKS:
        scrapper.link = link
        scrapper.findProducts(attribute='class', attributeDetail='ccm-core-commerce-add-to-cart', rootIncluded=False)

    scrapper.scrapeProductsMartin()

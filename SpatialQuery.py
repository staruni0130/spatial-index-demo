import time
import sys
import numpy as np
import sklearn
from sklearn.neighbors import KDTree
import pandas as pd
from pandas import read_csv
import shapely
from shapely.geometry import Polygon, Point, box
import rtree
import geopandas as gpd
import re


#-------------------------------task 1---------------------------------

def task_1(dataset_dir, testset_dir):
    print("Task 1 processing...")
    
    df = read_csv(dataset_dir)
    features = ['longitude', 'latitude']
    ################################################################
    #KDTree Indexing#
    index_start = time.time()

    kd_tree = KDTree(df[features], leaf_size=df[features].shape[0]+1)

    print('indexing time:', time.time() - index_start)
    ################################################################

    ################################################################
    #Reading test points from test file
    #And query points
    total_query = time.time()
    with open(testset_dir) as file:
        for line in file:
            #format test points
            patn = re.sub(r"[\([{})\]]", "", line)          
            patn = patn.split()
            
            #set test point
            point=[[float(patn[0]),float(patn[1])]]
            #set K
            K=int(patn[2])

            query_start = time.time()
            #query kdtree      
            distances, ndx = kd_tree.query(point, K, return_distance=True)
            print('query time:', time.time() - query_start)
            
            #get result by index from kdtree
            result=df.loc[ndx[0]]
            result['dist']=distances[0,:]
            #sort by id in case same distance
            result=result.sort_values(['dist','id'],ascending=[True,True])
            #print(result['id'])
            
            #write file
            result['id'].to_csv('task1_results.txt', header=None, index=None, sep='\t', mode='a')
    ################################################################
    print('Total query time:', time.time() - total_query)
    print("Task 1 done")

#-------------------------------task 2---------------------------------

def task_2(dataset_dir, testset_dir):

    print("Task 2 processing...")
    total_excute = time.time()

    df = read_csv(dataset_dir)

    features = ['longitude', 'latitude']
    ################################################################
    #Reading each test case from test file
    #  
    with open(testset_dir) as file:
        for line in file:
            
            #Format and get point, k, and datetime
            patn = re.sub(r"[\([{})\]]", "", line)
            get_time = re.findall(r'"([^"]*)"', line)
            start_time=pd.to_datetime(get_time[0])
            #print(start_time)
            end_time=pd.to_datetime(get_time[1])
            #print(end_time)
            patn = patn.split()
            point=[[float(patn[0]),float(patn[1])]]
            #print(point)
            K=int(patn[2])
            #print(K)
            
            #pre clean dataset by given datetime
            filtered_dataset = df
            filtered_dataset['date_time'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            filtered_dataset = filtered_dataset[(filtered_dataset['date_time'] >= start_time) & (filtered_dataset['date_time'] <= end_time)]
            filtered_dataset.reset_index(inplace=True, drop=True)
            
            ################################################################
            #KDTree Indexing#
            index_start = time.time()
            kd_tree = KDTree(filtered_dataset[features],leaf_size = filtered_dataset[features].shape[0]+1)
            print('indexing time:', time.time() - index_start)
            ################################################################
            
            ################################################################
            #query
            query_start= time.time()
            distances, ndx = kd_tree.query(point, K, return_distance=True)
            print("query time:", time.time() - query_start)
            ################################################################

            #get result and sort
            result=filtered_dataset.loc[ndx[0]]
            result['dist']=distances[0,:]
            result=result.sort_values(['dist','id'],ascending=[True,True])
            #print(result['id'])
            result['id'].to_csv('task2_results.txt', header=None, index=None, sep='\t', mode='a')


    print('Total excute time:', time.time() - total_excute)
    print("Task 2 done")

#-------------------------------task 3---------------------------------

def task_3(dataset_dir, testset_dir):
    print("Task 3 processing...")
    
    rectangle=[]
    ########################################################################
    #Format and get rectangles
    with open(testset_dir) as file:
        for line in file:
            patn = re.sub(r"[\([{})\]]", "", line)
            x = patn.split()
            rectangle.append(box(float(x[0]), float(x[1]), float(x[2]),float(x[3])))
    ########################################################################

    ########################################################################
    #Indexing
    index_start = time.time()
    idx = rtree.index.Index()
    for pos, poly in enumerate(rectangle):
        idx.insert(pos, poly.bounds)
    print('indexing time:', time.time() - index_start)
    #########################################################################

    #########################################################################
    #get points from dataset
    data = pd.read_csv(dataset_dir)
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))
    points = [(x,y) for x,y in zip(gdf['geometry'].x , gdf['geometry'].y)]
    #########################################################################

    #########################################################################
    #iterate each point and query
    query_start = time.time()
    f = open("task3_results.txt", "a")
    for k in enumerate(rectangle):
        for i,pt in enumerate(points):
            point=Point(pt)
            #check rtree index first
            if(idx.intersection(point.bounds)):
                #check if within given rectangle including on boundaries.
                if(point.intersects(rectangle[k[0]])):
                    result = gdf.loc[i]
                    #write result
                    f.write(str(result['id']))
                    f.write("\n")
    f.close()
    #########################################################################

    print('query time:', time.time() - query_start)    
    print("Task 3 done")

#-------------------------------task 4---------------------------------

def task_4(dataset_dir, testset_dir):

    print("Task 4 processing...")

    rectangle=[]
    start_time=[]
    end_time=[]

    #########################################################################
    #get rectangle and time window
    with open(testset_dir) as file:
        for line in file:
            patn = re.sub(r"[\([{})\]]", "", line)
            get_time=re.findall(r'"([^"]*)"', line)
            start_time.append(pd.to_datetime(get_time[0]))
            #print(start_time)
            end_time.append(pd.to_datetime(get_time[1]))
            #print(end_time)
            x = patn.split()
            rectangle.append(box(float(x[0]), float(x[1]), float(x[2]),float(x[3])))
    
    #print(start_time, end_time)
    #########################################################################

    #########################################################################
    #Indexing
    index_start = time.time()
    idx = rtree.index.Index()
    for pos, poly in enumerate(rectangle):
        idx.insert(pos, poly.bounds)
    print('indexing time:', time.time() - index_start)
    #########################################################################

    #########################################################################
    #get points from dataset
    data = pd.read_csv(dataset_dir)
    data['date_time'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))
    points = [(x,y) for x,y in zip(gdf['geometry'].x , gdf['geometry'].y)]
    #########################################################################


    #########################################################################
    #query and write result
    query_start = time.time()
    f = open("task4_results.txt", "a")
    for k in enumerate(rectangle):
        for i,pt in enumerate(points):
            #print(pt)
            point=Point(pt)
            if(idx.intersection(point.bounds)):
                if(point.intersects(rectangle[k[0]])):
                    result = gdf.loc[i]
                    #check time window
                    if(result['date_time']>=start_time[k[0]] and result['date_time']<=end_time[k[0]]):
                        #print("~~~ok")
                        f.write(str(result['id']))
                        f.write("\n")
                        #print('\n================================')
    f.close()
    #########################################################################
    print('query time:', time.time() - query_start)   
    print("Task 4 done")

#-------------------------------task 5---------------------------------

def task_5(dataset_dir, testset_dir):

    print("Task 5 processing...")

    df = read_csv(dataset_dir)
    features = ['longitude', 'latitude']
    ################################################################
    #KDTree Indexing#
    index_start = time.time()

    kd_tree = KDTree(df[features], leaf_size=df[features].shape[0]+1)

    print('indexing time:', time.time() - index_start)
    #################################################################

    #################################################################
    with open(testset_dir) as file:
        for line in file:
            #Format and get points, d, and date
            patn = re.sub(r"[\([{})\]]", "", line)
            get_time = re.findall(r'"([^"]*)"', line)
            start_time=pd.to_datetime(get_time[0])
            print(start_time)
            patn = patn.split()
            d = float(patn[-1])/100
            #get points,
            patn = patn[0:-2] 
            points = []
            for i in range(0,len(patn),2):
                points.append([patn[i], patn[i+1]])
            
            ################################################################
            #query
            id = []
            query_start = time.time()
            result,dist = kd_tree.query_radius(points, d, True)
            print("query time:", time.time() - query_start)
            ################################################################

            ################################################################
            #check date
            for i in range(len(result)):
                for j in range(0,len(result[i])):
                    if(pd.to_datetime(df.loc[result[i][j]]['date']) == start_time):
                        id.append(df.loc[result[i][j]]['id'])
            id.sort()                   # ascending sort
            id=list(dict.fromkeys(id))  # delete duplicates
            #################################################################

            #################################################################
            #write result
            f = open("task5_results.txt", "a")
            for i in range(len(id)):
                f.write(str(id[i]))
                f.write("\n")
            f.close()
            #################################################################
    print("Task 5 done")

#-------------------------------main entry-----------------------------

def main():
    if len(sys.argv)<2:
        print ("Error: forgot to include the directory name on the command line.")
        print ("Usage: python %s" % sys.argv[0])
        sys.exit()
    else:
        dataset_dir = sys.argv[1]
        task_no = sys.argv[2]
        testset_dir = sys.argv[3]

    #print("dataset_dir: " + dataset_dir + " \ntask_no: " + task_no + " \ntestset_dir: " + testset_dir)

    if(task_no == '1'):
        task_1(dataset_dir, testset_dir)
    if(task_no == '2'):
        task_2(dataset_dir, testset_dir)
    if(task_no == '3'):
        task_3(dataset_dir, testset_dir)
    if(task_no == '4'):
        task_4(dataset_dir, testset_dir)
    if(task_no == '5'):
        task_5(dataset_dir, testset_dir)


if __name__ == '__main__':
    main()

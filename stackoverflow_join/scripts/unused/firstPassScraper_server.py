import urllib2
import json
import time
import sys
import codecs
import pickle


template = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=categories&cllimit=max&titles=%s"
def titleFromLink(link):
    if link.find("<")==0: link = link[1:-1]
    return link.split("/")[-1].split("#")[0]
PREFIX="Category:"

def fetchCategoryData(query=None,title=None):
    if not query:
        query = template % title
    response = urllib2.urlopen(urllib2.Request(query,headers={'User-Agent':'GNAT/0.3 (krivard@cs.cmu.edu)'}))
    sdata = response.read()
    data = json.loads(sdata)
    return data

infile = 'fp_titles_outfile'
outfile = 'raw-firstpass_cats.tsv'

def scrape(infile,outfile,skip = 0):
    # not sure what "a" does
    with open(infile,"rb") as f,open(outfile,"w" if skip==0 else "a") as o:
        last = time.time()-1
        lastEntity = ""
        titles = pickle.load(f)
        for title in titles:
            #(tid,tname,wplink,meta) = line.strip().split("\t",3)
            now = title
            if now == lastEntity: continue
            lastEntity = now
            if skip>0: 
                skip = skip-1
                continue
            dt = time.time()-last
            if dt < 1: time.sleep(1-dt) # query at 1 Hz
            query = template % title
            print "%s %s ... " % (title,query),
            # try-except looping
            k = 1
            while True:
                try:
                    data = fetchCategoryData(query=query)
                    last = time.time()
                    n=0
                    if 'query' not in data or 'pages' not in data['query']:
                        print "Trouble with query '%s':" % title
                        print data
                        break
                    for page in data['query']['pages'].values():
                        if 'pageid' not in page: 
                            page['pageid'] = "N/A"
                            page['title'] = "N/A"
                        info = "#\t%s\n" % ("<" + page['title'] + ">")
                        o.write( info.encode("utf-8") )
                        if 'categories' not in page: continue
                        for cat in page['categories']:
                            foo = cat['title']
                            if foo.startswith(PREFIX): foo = foo[len(PREFIX):]
                            try:
                                o.write(foo.encode("utf-8"))
                                o.write("\n")
                            except UnicodeEncodeError:
                                print foo,type(foo)
                                raise
                            n+=1
                    print "%d categories" % n
                    break
                except:
                    print 'Network error with %s, trying again in %d seconds' %(title, k)
                    time.sleep(k)
                    k = 2*k
            
            

scrape(infile,outfile)
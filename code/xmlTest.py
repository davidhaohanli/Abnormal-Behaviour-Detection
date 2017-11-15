from xmlLoader_generator import *

thisPoi = Poi_handle()
print(tostring(thisPoi.tree, pretty_print=True).decode('utf-8'))
print(thisPoi.tree.getroot()[0].get('n'))
thisPoi.add(3,100,53,64)
thisPoi.add(4,213,124,1234)
thisPoi.add(3,100,23,623242345234)
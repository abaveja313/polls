from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
import matplotlib.colors as mcol

# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
# draw state boundaries.
# data from U.S Census Bureau
# http://www.census.gov/geo/www/cob/st2000.html
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)
# population density by state from
# http://en.wikipedia.org/wiki/List_of_U.S._states_by_population_density
popdensity = {
'New Jersey':  0.0,
'Rhode Island':   1.0,
'Massachusetts':   1.0,
'Connecticut':	  0.0,
'Maryland':   0.0,
'New York':    1.0,
'Delaware':    11.0,
'Florida':     1.0,
'Ohio':	 0.0,
'Pennsylvania':	 0.0,
'Illinois':    1.0,
'California':  1.0,
'Hawaii':  0.0,
'Virginia':  0.0,
'Michigan':    1.0,
'Indiana':    1.0,
'North Carolina':  1.0,
'Georgia':    0.0,
'Tennessee':  1.0,
'New Hampshire':  0,
'South Carolina':  1,
'Louisiana':  1,
'Kentucky':   1,
'Wisconsin':0,
'Washington': 0,
'Alabama':     1,
'Missouri': 1,
'Texas':   1,
'West Virginia':   0,
'Vermont':    0,
'Minnesota':  1,
'Mississippi': 1,
'Iowa':	 0,
'Arkansas':   0,
'Oklahoma':    1,
'Arizona':   1,
'Colorado':    0,
'Maine':  0,
'Oregon':  1,
'Kansas':  1,
'Utah':	 1,
'Nebraska':   1,
'Nevada': 1,
'Idaho': 1  ,
'New Mexico':  0,
'South Dakota':	 0,
'North Dakota':	 0,
'Montana':     0,
'Wyoming':      1,
'Alaska':     1}
print(shp_info)
# choose a color for each state based on population density.
colors={}
statenames=[]
cmap = plt.cm.jet # use 'hot' colormap
vmin = 0; vmax = 1 # set range.
print(m.states_info[0].keys())
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia','Puerto Rico']:
        pop = popdensity[statename]
        # calling colormap with value between 0 and 1 returns
        # rgba value.  Invert color range (hot colors are high
        # population), take sqrt root to spread out colors more.
        colors[statename] = cmap(1.-np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
    statenames.append(statename)
# cycle through state names, color each one.
ax = plt.gca() # get current axes instance
for nshape,seg in enumerate(m.states):
    # skip DC and Puerto Rico.
    if statenames[nshape] not in ['District of Columbia','Puerto Rico']:
        color = rgb2hex(colors[statenames[nshape]]) 
        poly = Polygon(seg,facecolor=color,edgecolor=color)
        ax.add_patch(poly)
# draw meridians and parallels.
m.drawparallels(np.arange(25,65,20),labels=[1,0,0,0])
m.drawmeridians(np.arange(-120,-40,20),labels=[0,0,0,1])
plt.title('Filling State Polygons by Population Density')
plt.show()

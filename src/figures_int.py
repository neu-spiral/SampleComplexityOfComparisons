import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.path
import matplotlib.patches as patches

#         FancyArrowPatch.draw(self, renderer)

##############################################################################################################################

color = sns.color_palette("Paired")
# blue (light, dark), red (light, dark), green (light, dark), orange (light, dark), purple (light, dark), brown
# blue and orange are color blind friendly.
d = 2
n = 200
np.random.seed(6)
beta = 4*np.random.randn(2,1)
np.random.seed(2)
points = np.random.randn(n,2)

dot_products = np.dot(points,beta)
ind_pos_pts = np.where(dot_products>=0)[0]
ind_neg_pts = np.where(dot_products<0)[0]

# xx, yy = np.meshgrid(np.linspace(min(points[:,0]),max(points[:,0]),5),np.linspace(min(points[:,1]),max(points[:,1]),5))
x = np.linspace(-3.5,3.5,5)
# calculate corresponding z
slope_of_beta = 1.*beta[1,0]/beta[0,0]
y = -1./slope_of_beta*x
beta_est = 1.*(np.sum(points[ind_pos_pts,:],axis=0) -np.sum(points[ind_neg_pts,:],axis=0))/n

beta_est = 1.5*beta_est
##############################################################################################################################
# Fig 1

fig1 = plt.figure()
# ax = fig1.add_subplot(111)
plt.scatter(points[ind_pos_pts,0],points[ind_pos_pts,1],c=color[1],marker='^',s=30, label='Positives')
plt.scatter(points[ind_neg_pts,0],points[ind_neg_pts,1],c=color[7],marker='s',s=30, label='Negatives')
plt.annotate(r'', xy=(0, 0),  size= 15,xycoords='data',
            xytext=(beta[0,0],beta[1,0]),arrowprops=dict(arrowstyle="<-",facecolor='black'))
plt.text(-1.3,3,r"$\beta$",fontsize=20)
# plt.quiver(0,0,beta[0,0],beta[1,0],scale=2,scale_units='xy')
# ind_pos_select = 1
pos_pt = [-2.45,1.75]
neg_pt = [-2,-1.4]
plt.annotate(r'', xy=(0, 0), size= 15, xycoords='data',
            xytext=(-2.45,1.75),arrowprops=dict(arrowstyle="<-",facecolor='black'))
plt.text(-2.4,1.9,r"$P_{1}$",fontsize=20)
# ind_neg_select = 30 #13, 25
plt.annotate(r'', xy=(0, 0), size=15, xycoords='data',
            xytext=(-1.45,-2.05),arrowprops=dict(arrowstyle="<-",facecolor='black'))
plt.text(-2,-1.8,r"$P_{2}$",fontsize=20)

plt.axis('square')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.plot(x,y,color='k')
plt.legend(fontsize=15,loc='lower left')
# plt.show()
fig1.savefig("../fig/data_origin.pdf", bbox_inches='tight')


##############################################################################################################################
# Fig 2

fig2 = plt.figure()
plt.scatter(points[ind_pos_pts,0],points[ind_pos_pts,1],c=color[1],marker='^',s=30, label='Positives')
plt.scatter(-points[ind_neg_pts,0],-points[ind_neg_pts,1],c=color[7],marker='s',s=30, label='Negatives (Mirrored)')
plt.annotate(r'', xy=(0, 0),  size= 15,xycoords='data',
            xytext=(beta[0,0],beta[1,0]),arrowprops=dict(arrowstyle="<-",facecolor='black'))
plt.text(-1.3,3,r"$\beta$",fontsize=20)
# plt.quiver(0,0,beta[0,0],beta[1,0],scale=2,scale_units='xy')
# ind_pos_select = 1
plt.annotate(r'', xy=(0, 0), size= 15, xycoords='data',
            xytext=(-2.45,1.75),arrowprops=dict(arrowstyle="<-",facecolor='black'))
plt.text(-2.4,1.9,r"$P_{1}^{'}$",fontsize=20)
# ind_neg_select = 30 #13, 25
plt.annotate(r'', xy=(0, 0), size=15, xycoords='data',
            xytext=(1.5,2.05),arrowprops=dict(arrowstyle="<-",facecolor='black'))
plt.text(1.4,1.5,r"$P_{2}^{'}$",fontsize=20)
plt.axis('square')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.plot(x,y,color='k')
plt.legend(fontsize=15)
# plt.show()
fig2.savefig("../fig/data_mirrored.pdf", bbox_inches='tight')


##############################################################################################################################
# Fig 3
# plot the surface
fig3 = plt.figure()
plt.scatter(points[ind_pos_pts,0],points[ind_pos_pts,1],c=color[1],marker='^',s=30, label='Positives')
plt.scatter(-points[ind_neg_pts,0],-points[ind_neg_pts,1],c=color[7],marker='s',s=30, label='Negatives (Mirrored)')
plt.annotate(r'', xy=(0, 0),  size= 15,xycoords='data',
            xytext=(beta[0,0],beta[1,0]),arrowprops=dict(arrowstyle="<-",facecolor='black'))
plt.text(-1.3,3,r"$\beta$",fontsize=20)
plt.annotate(r'$\hat{\beta}$',
            xy=(-0.4, 0.6), xycoords='data', size=20,
            xytext=(0, -1),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"))
# plt.text(-1.3,3,r"$\beta$",fontsize=13)
plt.annotate(r'', xy=(0, 0),  size= 15,xycoords='data',
            xytext=(1.1*beta_est[0],1.1*beta_est[1]),arrowprops=dict(arrowstyle="<|-",facecolor='black',linewidth=5))
# plt.quiver(0,0,beta[0,0],beta[1,0],scale=2,scale_units='xy')

plt.annotate(r'', xy=(0, 0), size= 15, xycoords='data',
            xytext=(-2.45,1.75),arrowprops=dict(arrowstyle="<-",facecolor='black'))
plt.text(-2.4,1.9,r"$P_{1}^{'}$",fontsize=20)
# ind_neg_select = 30 #13, 25
plt.annotate(r'', xy=(0, 0), size=15, xycoords='data',
            xytext=(1.5,2.05),arrowprops=dict(arrowstyle="<-",facecolor='black'))
plt.text(1.4,1.5,r"$P_{2}^{'}$",fontsize=20)
plt.axis('square')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.plot(x,y,color='k')
plt.legend(fontsize=15)
plt.show()
fig3.savefig("../fig/data_mirrored_est.pdf", bbox_inches='tight')
print("done")









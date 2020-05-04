import numpy as np
import matplotlib as mpl
mpl.use('Agg')

# Following code block from https://matplotlib.org/tutorials/text/pgf.html
pgf_with_custom_preamble = {
    "font.family": "serif", # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    # "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    "pgf.preamble": [
         "\\usepackage{unicode-math,amsmath,amssymb,amsthm}",  # unicode math setup
         ]
}
mpl.rcParams.update(pgf_with_custom_preamble)

import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def makeup_for_plot(fig1):
	fig1.spines["top"].set_visible(False)    
	fig1.spines["bottom"].set_visible(True)    
	fig1.spines["right"].set_visible(False)    
	fig1.spines["left"].set_visible(True)
	fig1.get_xaxis().tick_bottom()    
	fig1.get_yaxis().tick_left()
	fig1.tick_params(axis="both", which="both", bottom="off", top="off",    
			labelbottom="on", left="off", right="off", labelleft="on",labelsize=12)
	grid_color = '#e3e3e3'
	grid_line_style= '--'
	fig1.grid(linestyle=grid_line_style,color=grid_color)
	return fig1

def do_tight_layout_for_fig(fig):
	fig.tight_layout()
	return fig

lr_vals = [0.1]

colors = ['red','green','c','m','y','orange','green']

import argparse
parser = argparse.ArgumentParser(description='Plots')
parser.add_argument('--fun_num', '--fun_num', default=0,type=int,  dest='fun_num')
args = parser.parse_args()
fun_num = args.fun_num

# TODO: fun_num here is different from the generate_results.sh
# will make it consistent later. 

my_markers = ['','','','','','','']


if fun_num == 0:
	# for L2 Regularization for U,Z and lam = 0
	files = {
		1: 'results/cocain_1_abs_fun_num_1.txt',
		2: 'results/gd_bt_1_abs_fun_num_1.txt',
		3: 'results/gd_bt_global_1_abs_fun_num_1.txt',
		4: 'results/ibgm_1_abs_fun_num_1.txt',
		5: 'results/cocain_cf_1_abs_fun_num_1.txt',
	}
if fun_num == 1:
	files = {
		1: 'results/cocain_2_abs_fun_num_2.txt',
		2: 'results/gd_bt_2_abs_fun_num_2.txt',
		3: 'results/gd_bt_global_2_abs_fun_num_2.txt',
		4: 'results/ibgm_2_abs_fun_num_2.txt',
		5: 'results/cocain_cf_2_abs_fun_num_2.txt',
	}


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1 = makeup_for_plot(ax1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2 = makeup_for_plot(ax2)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3 = makeup_for_plot(ax3)
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4 = makeup_for_plot(ax4)


label_font_size = 13
legend_font_size = 17
my_line_width = 2



labels_dict = {
		1: r"CoCaIn BPG",
		2: r"BPG-WB", 
		3: r"BPG",
		4: r"IBPM-LS",
		5: r"CoCaIn BPG CFI"
		}

nb_epoch = 1000
opt_vals= np.array([2,3,4])

color_count = 0

f_opt = 0


min_fun_val = np.inf
for i in opt_vals:
	file_name = files[i]
	try:
		best_train_objective_vals = np.loadtxt(file_name)[:,0]
		min_fun_val = np.nanmin([min_fun_val,np.min(best_train_objective_vals)])
		print('Min function val is '+ str(min_fun_val))
	except:
		pass

for i in opt_vals:
	file_name = files[i] 
	print(file_name)
	try:
		if i in [1,5]:
			best_train_objective_vals = np.loadtxt(file_name)[:,0]
			best_lb_est_vals = np.loadtxt(file_name)[:,3]
			best_gamma_est_vals = np.loadtxt(file_name)[:,4]
			best_time_vals = np.loadtxt(file_name)[:,5]
		else:
			best_train_objective_vals = np.loadtxt(file_name)[:,0]
			best_time_vals = np.loadtxt(file_name)[:,1]
	except:
		print("using thiss")
		best_train_objective_vals = np.loadtxt(file_name)

	print(fun_num)

	ax1.loglog((np.arange(nb_epoch)+1),(best_train_objective_vals[:nb_epoch]),\
			label=labels_dict[i],color=colors[color_count],\
			linewidth=my_line_width,marker=my_markers[i-1])
	ax4.loglog((np.arange(nb_epoch)+1), \
			(best_train_objective_vals[:nb_epoch] - min_fun_val),
            label=labels_dict[i], color=colors[color_count],
            linewidth=my_line_width, marker=my_markers[i-1])
	best_time_vals[0] = 1e-2

	ax2.loglog(np.cumsum(best_time_vals[:nb_epoch]), \
		(best_train_objective_vals[:nb_epoch]),\
		label=labels_dict[i],color=colors[color_count],\
		linewidth=my_line_width,marker=my_markers[i-1])

	if i in [1,5]:
		ax3.loglog((np.arange(nb_epoch)+1), (best_gamma_est_vals[:nb_epoch]),
				label=labels_dict[i], color=colors[color_count], \
				linewidth=my_line_width, marker=my_markers[i-1])

	color_count +=1

figure_name1 = 'figures/'+'func_vals_fun_num_'+str(fun_num)

# legends
ax1.legend(loc='upper right', fontsize=label_font_size)
ax2.legend(loc='upper right', fontsize=label_font_size)
ax3.legend(loc='upper right', fontsize=label_font_size)
ax4.legend(loc='upper right', fontsize=label_font_size)

ax1.set_xlabel('Iterations (log scale)',fontsize=legend_font_size)
ax1.set_ylabel('Function value (log scale)',fontsize=legend_font_size)


do_tight_layout_for_fig(fig1)
fig1.savefig(figure_name1+'.png', dpi=fig1.dpi)
fig1.savefig(figure_name1+'.pdf', dpi=fig1.dpi)

ax2.set_xlabel('Time (log scale)',fontsize=legend_font_size)
ax2.set_ylabel('Function value (log scale)',fontsize=legend_font_size)

do_tight_layout_for_fig(fig2)
fig2.savefig(figure_name1+'_time_.png', dpi=fig2.dpi)
fig2.savefig(figure_name1+'_time_.pdf', dpi=fig2.dpi)


ax3.set_xlabel('Iterations (log scale)', fontsize=legend_font_size)
ax3.set_ylabel('Gamma (log scale)', fontsize=legend_font_size)

print("Doing fig3")
do_tight_layout_for_fig(fig3)
fig3.savefig(figure_name1+'_gamma_.png', dpi=fig3.dpi)
fig3.savefig(figure_name1+'_gamma_.pdf', dpi=fig3.dpi)


ax4.set_xlabel('Iterations (log scale)', fontsize=legend_font_size)
ax4.set_ylabel('Suboptimality (log scale)', fontsize=legend_font_size)


do_tight_layout_for_fig(fig4)
fig4.savefig(figure_name1+'_suboptimality_.png', dpi=fig4.dpi)
fig4.savefig(figure_name1+'_suboptimality_.pdf', dpi=fig4.dpi)

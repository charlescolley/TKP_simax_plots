
#from sqlalchemy import true
from sqlalchemy import false
import CTAMELogParser as CT_LP

from plotting_style import *


def call_all_plots():
    #
    #  -- Low Rank Structure -- #
    #
    max_rank_experiments()
    TAME_vs_LRTAME_clique_scaling_summarized()

    # Appendix plots
    TAME_vs_LRTAME_rank_1_case_singular_values()

    # Supplementary File
    TAME_vs_LRTAME_clique_scaling_detailed() 
    

    #
    #  -- Random Graph Experiments --  #
    #
    RandomGeometricRG_PostProcessing_is_needed()
    LambdaTAME_increasing_clique_size_v2()
    

    #Supplementary Plots 
    RandomGeometricDupNoise_allModes()
    RandomGeometricERNoise_allModes()
        # NOTE: default params are n = 250, p = .05


    #
    #  -- LVGNA Experiments --  #
    #
    LVGNA_end_to_end_relative_to_TAME_table_with_microplots()
    make_LVGNA_TTVMatchingRatio_runtime_plots()

    # Supplementary figures 
    LVGNA_pre_and_post_processed()

    #
    #  -- Dominant Eigenpair Experiments --  #
    #
    tensor_kronecker_product_eigenspaces_as_row()

def render_and_save_all_plots(output_path=None):
    if output_path is None:
        output_path = "../rendered_figures/"
    assert output_path[-1] == "/"
    make_path = lambda filename: output_path + filename
    #
    #  -- Low Rank Structure -- #
    #
    max_rank_experiments(save_path=make_path("MaxRankExperiments.pdf"))
    print(f"saved fig {make_path('MaxRankExperiments.pdf')}")
    TAME_vs_LRTAME_clique_scaling_summarized(save_path=make_path("TAME_LRTAME_clique_scaling_summarized.pdf"))
    print(f"saved fig {make_path('TAME_LRTAME_clique_scaling_summarized.pdf')}")
 
    # Appendix plots
    TAME_vs_LRTAME_rank_1_case_singular_values(save_path=make_path("LRTAME_vs_TAME_rank_1_case_secondSingularValues.pdf"))
    print(f"saved fig {make_path('LRTAME_vs_TAME_rank_1_case_secondSingularValues')}")
  
    # Supplementary File
    TAME_vs_LRTAME_clique_scaling_detailed(save_path=make_path("TAME_LRTAME_clique_scaling.pdf"))
        #TODO: need to update with 8 + 9 Clique results
        #      violin plots style need to be updated too. 
    print(f"saved fig {make_path('TAME_LRTAME_clique_scaling.pdf')}")
    


    #
    #  -- Random Graph Experiments --  #
    #
    RandomGeometricRG_PostProcessing_is_needed(save_path=make_path('SynthAlignments_SizeExps_theNeedForPostProcessing.pdf'))
    print(f"saved fig {make_path('SynthAlignments_SizeExps_theNeedForPostProcessing.pdf')}")

    LambdaTAME_increasing_clique_size_v2(save_path=make_path('PostProcessing_KNearest_IncreasingMotifs.pdf'))
    print(f"saved fig {make_path('PostProcessing_KNearest_IncreasingMotifs.pdf')}")
    
    
    #Supplementary Plots 
    RandomGeometricDupNoise_allModes(save_path=make_path('DupNoiseRG_exps.pdf'))
    print(f"saved fig {make_path('DupNoiseRG_exps.pdf')}")
    RandomGeometricERNoise_allModes(save_path=make_path('ERNoiseRG_exps.pdf'))
    print(f"saved fig {make_path('ERNoiseRG_exps.pdf')}")
        # TODO: add in LREA post processing results 
        #       update plot style 
        # NOTE: default params are n = 250, p = .05


    #
    #  -- LVGNA Experiments --  #
    #
    LVGNA_end_to_end_relative_to_TAME_table_with_microplots(save_path=make_path('LVGNA_end_to_end_vs_TAME.pdf'))
    print(f"saved fig {make_path('LVGNA_end_to_end_vs_TAME.pdf')}")
    
    make_LVGNA_TTVMatchingRatio_runtime_plots(save_path=make_path('LVGNA_TTVMatchingRatio_noLowRank.pdf'))
    print(f"saved fig {make_path('LVGNA_TTVMatchingRatio_noLowRank.pdf')}")
    # Supplementary figures 
    LVGNA_pre_and_post_processed(save_path=make_path('TAME_LVGNA_PreAndPostProcessing.pdf'))
    print(f"saved fig {make_path('TAME_LVGNA_PreAndPostProcessing.pdf')}")

    #
    #  -- Dominant Eigenpair Experiments --  #
    #
    tensor_kronecker_product_eigenspaces_as_row(save_path=make_path('TKPDominantEigenpairs.pdf'))
    print(f"saved fig {make_path('TKPDominantEigenpairs.pdf')}")

#
#   Matlab Experiment Summary stats
#

def verify_ARST_tensors_are_generated_tensors():
    """ Copies of the tensors are saved within the AReigST files too.
        This function verifies all the tensors are the same to ensure
        correct spectra are used across all algorithms."""
    
    tensor_path = TKP_RESULTS + "tensor_problems/"
    tensors =  {}

    for file in os.listdir(tensor_path):
        tenParamStr = file.split(".mat")[0].split("weighted_TKP_")[-1]
        mat_file_obj = scipy.io.loadmat(tensor_path+file)
        
        A = mat_file_obj["A"][0][0][0]
        B = mat_file_obj["B"][0][0][0]
        A_kron_B = mat_file_obj["A_kron_B"][0]

        tensors[tenParamStr] = [A,B,A_kron_B]

    ARST_result_path = TKP_RESULTS + "AREigST/"

    for file in os.listdir(ARST_result_path):
        tenParamStr = file.split("_results.mat")[0].split("weighted_kronzeig_exp_")[-1]
        mat_file_obj = scipy.io.loadmat(ARST_result_path+file)

        A = mat_file_obj["A"][0][0][0]
        B = mat_file_obj["B"][0][0][0]
        A_kron_B = mat_file_obj["A_kron_B"][0]

        A_prob, B_prob, A_kron_B_prob = tensors[tenParamStr]

        print(f"file:{file}\n |A-Ap|={np.sum(A - A_prob)}  |B -Bp|={np.sum(B - B_prob)}  |A_kron_B -A_kron_Bp|={np.sum(A_kron_B - A_kron_B_prob)}")

def tensor_kronecker_product_eigenspaces(save_path=None):


    #
    #   Data loading drivers
    #

    def process_AReigSTdata():

        result_path = TKP_RESULTS + "AReigST/"

        #
        #   Load data from files
        #

        success_codes = []

        eigenvectors_found = {}

        for file in os.listdir(result_path):


            tensor_param_str = file.split("_results.mat")[0].split("exp_")[-1]
            mat_file = scipy.io.loadmat(result_path + file)

            codes = []

            for key in ["info_B","info_A","info_A_kron_B"]:
                codes.append(mat_file[key][0]["success"][0][0][0])

            A_eigvals = mat_file["lmd_A"].reshape(-1)
            B_eigvals = mat_file["lmd_B"].reshape(-1)
            A_kron_B_eigvals = mat_file["lmd_A_kron_B"].reshape(-1)

            A_eigvecs  = mat_file["eigvec_A"]          
            B_eigvecs  = mat_file["eigvec_B"]
            A_kron_B_eigvecs = mat_file["eigvec_A_kron_B"]

            eigenvectors_found[tensor_param_str] = [A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, A_kron_B_eigvals, A_kron_B_eigvecs]
            success_codes.append(codes)

        return eigenvectors_found,success_codes

    def process_NCMdata(subdirectory="NCM_sampling/"):

        results_path = TKP_RESULTS +subdirectory
        eigenvectors_found = {}

        for file in os.listdir(results_path):

            tensor_param_str = file.split("_delta:")[0].split("NCM_")[-1]

            matfile_object = scipy.io.loadmat(results_path+file)

            A_eigvals, A_eigvecs, _,_ = matfile_object["A_output"][0][0]
            B_eigvals, B_eigvecs,  _,_= matfile_object["B_output"][0][0]
            A_kron_B_eigvals, A_kron_B_eigvecs, _,_ = matfile_object["A_kron_B_output"][0][0]


            A_eigvecs = np.transpose(A_eigvecs)
            B_eigvecs = np.transpose(B_eigvecs)
            A_kron_B_eigvecs = np.transpose(A_kron_B_eigvecs)
    
            A_eigvals = A_eigvals.reshape(-1)
            B_eigvals = B_eigvals.reshape(-1)
            A_kron_B_eigvals = A_kron_B_eigvals.reshape(-1)


            eigenvectors_found[tensor_param_str] = [A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, A_kron_B_eigvals, A_kron_B_eigvecs]


        return eigenvectors_found
    
    AReigST_codes = process_AReigSTdata()[1]
    eigenpairs_found = [
        process_AReigSTdata()[0],
        process_NCMdata(subdirectory="NCM_sampling/"),
        process_NCMdata(subdirectory="ONCM_sampling/"),
    ]
    #return eigenpairs_found

    #
    #   Consolidate the eigevectors found for each tensor 
    #
    def update_max(eigvals,eigvec,curr_eigpair):
        # curr_eigpair::(eigval,eigvec)
        vals = abs(eigvals)
        idx = np.argmax(vals)
        #print(vals)
        lmd = vals[idx]


        if lmd > curr_eigpair[0]:
            return (lmd,eigvec[idx,:]), True
        else:
            return curr_eigpair, False


    dominant_eigenpairs = []
    algorithm_with_best_results = []
    for tenParamStr in eigenpairs_found[0].keys():


        max_A_eigpair = (-np.Inf,[])
        max_B_eigpair = (-np.Inf,[])
        max_A_kron_B_eigpair = (-np.Inf,[])
        
        algorithm_chosen = [-1,-1,-1]
                           # 3 entries for A,B,A_kron_B


        for alg_idx,exp in enumerate(eigenpairs_found):
            (A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, A_kron_B_eigvals, A_kron_B_eigvecs) = exp[tenParamStr]

            max_A_eigpair, was_updated = update_max(A_eigvals, A_eigvecs,max_A_eigpair) 
            if was_updated:
                algorithm_chosen[0] = alg_idx
             
            max_B_eigpair, was_updated = update_max(B_eigvals, B_eigvecs,max_B_eigpair)
            if was_updated:
                algorithm_chosen[1] = alg_idx
            
            max_A_kron_B_eigpair, was_updated = update_max(A_kron_B_eigvals, A_kron_B_eigvecs, max_A_kron_B_eigpair)
            if was_updated:
                algorithm_chosen[2] = alg_idx
            
        algorithm_with_best_results.append(algorithm_chosen)


        dominant_eigenpairs.append([max_A_eigpair,max_B_eigpair,max_A_kron_B_eigpair])
 
    lmd_diff = []
    subspace_angle = []


    for i,((A_eigpair,B_eigpair,A_kron_B_eigpair),file) in enumerate(zip(dominant_eigenpairs,eigenpairs_found[0].keys())):
        (A_val,A_vec) = A_eigpair
        (B_val,B_vec) = B_eigpair
        (A_kron_B_val,A_kron_B_vec) = A_kron_B_eigpair
            
    
        lmd_diff.append((A_val*B_val-A_kron_B_val)/A_kron_B_val)
        subspace_angle.append(1-abs(np.dot(np.kron(B_vec,A_vec),A_kron_B_vec)))

        print(f"{file}: diff:{lmd_diff[-1]}   angle:{subspace_angle[-1]}   ARST_codes:{AReigST_codes[i]}")


    #
    #   Plotting Subroutines
    #
 


    def make_violin_plot(ax,data,precision=2,c=None,v_alpha=.8,format="default",xlim=None,xscale="linear",column_type=None):

        if xscale=="linear":
            v = ax.violinplot(data,[.5], points=100, showmeans=False,widths=.15,
                        showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        elif xscale=="log":
            v = ax.violinplot(np.log10(data),[.5], points=100, showmeans=False,widths=.15,showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        else:
            raise ValueError(f"supports xscale values: 'linear' & 'log'; Got {xscale}.")


        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        newMedianLines = [[(x0,y1),(x0,y1+.7)]]
        v["cmedians"].set_segments(newMedianLines)

        # -- place extremal markers underneath
        extremal_tick_ypos = .25

        # -- write data values as text
        bbox = dict(boxstyle="round", ec="w", fc="w", alpha=.7,pad=.01)
        if column_type is None:

            if format == "default":
                ax.annotate(f"{np.median(data):.{precision}f}",xy=(.5,.4),xycoords="axes fraction",ha="center",fontsize=10).set_bbox(bbox)
                ax.annotate(f"{np.min(data):.{precision}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8).set_bbox(bbox)
                ax.annotate(f"{np.max(data):.{precision}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8).set_bbox(bbox)
            elif format == "scientific":
                ax.annotate(f"{np.median(data):.{precision}e}",xy=(.5,.4),xycoords="axes fraction",ha="center",fontsize=10).set_bbox(bbox)
                ax.annotate(f"{np.min(data):.{precision}e}",xy=(.025,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8).set_bbox(bbox)
                ax.annotate(f"{np.max(data):.{precision}e}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8).set_bbox(bbox)
            else:
                print(f"expecting format to be either 'default' or 'scientific', got:{format}")
        elif column_type == "merged_axis":
            pass
        else:
            raise ValueError("column_type expecting 'merged_axis' or None, but got {column_type}\n")

        if c is not None:
            v["cmedians"].set_color(c)
            v["cmedians"].set_alpha(.75)
            for b in v['bodies']:
                # set colors 
                b.set_facecolor("None")
                b.set_edgecolor(c)            
                b.set_alpha(v_alpha)

                #  -- only plot the top half of violin plot --  #
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)
                
                #  -- clip the top of the med-line to the top of the violin plot --  #
                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )

                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])

 
    def make_violin_plot_legend(ax,c="k"):
        
        np.random.seed(12)#Ok looking:12
        data = [np.random.normal() for i in range(50)]
        v = ax.violinplot(data, points=100, positions=[.6], showmeans=False, 
                        showextrema=False, showmedians=True,widths=.6,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #ax.axes.set_axis_off()
        ax.set_ylim(.5,1.0)
        ax.patch.set_alpha(0.0) 
                # turn off background
        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        newMedianLines = [[(x0,y1 +.05),(x0,1.5)]]
        v["cmedians"].set_segments(newMedianLines)

        ax.annotate("median",xy=(.5,.325),xycoords="axes fraction",ha="center",va="center",fontsize=10)
        ax.annotate(f"min",xy=(.025,-.075),xycoords="axes fraction",ha="left",fontsize=9,alpha=.8)
        ax.annotate(f"max",xy=(.975,-.075),xycoords="axes fraction",ha="right",fontsize=9,alpha=.8)

        if c is not None:
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor(c)
                b.set_alpha(.3)
                b.set_color(c)
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)

                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    new_max_y += .02
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])

    fig = plt.figure(figsize=(2.1,2))
    n = 1
    m = 2 # 4
    gs = fig.add_gridspec(m, 1,
                        left=0.025, right=0.975,top=.975,bottom=.2,
                        wspace=.05,hspace=.1)

    all_axes = np.empty(m,object)
    for j in range(m):
        all_axes[j] = fig.add_subplot(gs[j])
                
    replace_zeros_with_machine_epsilon = lambda vals: [2e-16 if x == 0 else abs(x) for x in vals]

    make_violin_plot(all_axes[0],replace_zeros_with_machine_epsilon(lmd_diff),xscale="log",c="k",format="scientific",precision=3)
    make_violin_plot(all_axes[1],replace_zeros_with_machine_epsilon(subspace_angle),xscale="log",c="k",format="scientific",precision=3)
    
    #
    #   Touch up Axes
    #

    legend_axis = all_axes[1].inset_axes([.2,-.5,.6,.5])
    make_violin_plot_legend(legend_axis)
    
    for ax in chain(all_axes.reshape(-1),[legend_axis]):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis="both",which='major', length=0,pad=6)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    #for ax in all_axes:
        #ax.set_ylim(.3,.8)
    n = len(all_axes.reshape(-1))
    for (i,ax) in zip(range(n-1,0,-1),all_axes.reshape(-1)):
        ax.set_zorder(i)

    all_axes[0].annotate(r"$||\lambda_B||\lambda_A| - |\lambda_{B \otimes A}||/|\lambda_{B \otimes A}|$",xy=(0.5 , 0.2), xycoords='axes fraction',ha="center",va="top",fontsize=12)
    all_axes[1].annotate(r"$1-|\langle {\bf v_{B}} \otimes {\bf v_{A}},{\bf v_{B \otimes A}}\rangle|$",xy=(0.5 , 0.2), xycoords='axes fraction',ha="center",va="top",fontsize=12)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def tensor_kronecker_product_eigenspaces_as_row(save_path=None):


    #
    #   Data loading drivers
    #

    def process_AReigSTdata():

        result_path = TKP_RESULTS + "AReigST/"

        #
        #   Load data from files
        #

        success_codes = []

        eigenvectors_found = {}

        for file in os.listdir(result_path):


            tensor_param_str = file.split("_results.mat")[0].split("exp_")[-1]
            mat_file = scipy.io.loadmat(result_path + file)

            codes = []

            for key in ["info_B","info_A","info_A_kron_B"]:
                codes.append(mat_file[key][0]["success"][0][0][0])

            A_eigvals = mat_file["lmd_A"].reshape(-1)
            B_eigvals = mat_file["lmd_B"].reshape(-1)
            A_kron_B_eigvals = mat_file["lmd_A_kron_B"].reshape(-1)

            A_eigvecs  = mat_file["eigvec_A"]          
            B_eigvecs  = mat_file["eigvec_B"]
            A_kron_B_eigvecs = mat_file["eigvec_A_kron_B"]

            eigenvectors_found[tensor_param_str] = [A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, A_kron_B_eigvals, A_kron_B_eigvecs]
            success_codes.append(codes)

        return eigenvectors_found,success_codes

    def process_NCMdata(subdirectory="NCM_sampling/"):

        results_path = TKP_RESULTS +subdirectory
        eigenvectors_found = {}

        for file in os.listdir(results_path):

            tensor_param_str = file.split("_delta:")[0].split("NCM_")[-1]

            matfile_object = scipy.io.loadmat(results_path+file)

            A_eigvals, A_eigvecs, _,_ = matfile_object["A_output"][0][0]
            B_eigvals, B_eigvecs,  _,_= matfile_object["B_output"][0][0]
            A_kron_B_eigvals, A_kron_B_eigvecs, _,_ = matfile_object["A_kron_B_output"][0][0]


            A_eigvecs = np.transpose(A_eigvecs)
            B_eigvecs = np.transpose(B_eigvecs)
            A_kron_B_eigvecs = np.transpose(A_kron_B_eigvecs)
    
            A_eigvals = A_eigvals.reshape(-1)
            B_eigvals = B_eigvals.reshape(-1)
            A_kron_B_eigvals = A_kron_B_eigvals.reshape(-1)


            eigenvectors_found[tensor_param_str] = [A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, A_kron_B_eigvals, A_kron_B_eigvecs]


        return eigenvectors_found
    
    AReigST_codes = process_AReigSTdata()[1]
    eigenpairs_found = [
        process_AReigSTdata()[0],
        process_NCMdata(subdirectory="NCM_sampling/"),
        process_NCMdata(subdirectory="ONCM_sampling/"),
    ]
    #return eigenpairs_found

    #
    #   Consolidate the eigevectors found for each tensor 
    #
    def update_max(eigvals,eigvec,curr_eigpair):
        # curr_eigpair::(eigval,eigvec)
        vals = abs(eigvals)
        idx = np.argmax(vals)
        #print(vals)
        lmd = vals[idx]


        if lmd > curr_eigpair[0]:
            return (lmd,eigvec[idx,:]), True
        else:
            return curr_eigpair, False


    dominant_eigenpairs = []
    algorithm_with_best_results = []
    for tenParamStr in eigenpairs_found[0].keys():


        max_A_eigpair = (-np.Inf,[])
        max_B_eigpair = (-np.Inf,[])
        max_A_kron_B_eigpair = (-np.Inf,[])
        
        algorithm_chosen = [-1,-1,-1]
                           # 3 entries for A,B,A_kron_B


        for alg_idx,exp in enumerate(eigenpairs_found):
            (A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, A_kron_B_eigvals, A_kron_B_eigvecs) = exp[tenParamStr]

            max_A_eigpair, was_updated = update_max(A_eigvals, A_eigvecs,max_A_eigpair) 
            if was_updated:
                algorithm_chosen[0] = alg_idx
             
            max_B_eigpair, was_updated = update_max(B_eigvals, B_eigvecs,max_B_eigpair)
            if was_updated:
                algorithm_chosen[1] = alg_idx
            
            max_A_kron_B_eigpair, was_updated = update_max(A_kron_B_eigvals, A_kron_B_eigvecs, max_A_kron_B_eigpair)
            if was_updated:
                algorithm_chosen[2] = alg_idx
            
        algorithm_with_best_results.append(algorithm_chosen)


        dominant_eigenpairs.append([max_A_eigpair,max_B_eigpair,max_A_kron_B_eigpair])
 
    lmd_diff = []
    subspace_angle = []


    for i,((A_eigpair,B_eigpair,A_kron_B_eigpair),file) in enumerate(zip(dominant_eigenpairs,eigenpairs_found[0].keys())):
        (A_val,A_vec) = A_eigpair
        (B_val,B_vec) = B_eigpair
        (A_kron_B_val,A_kron_B_vec) = A_kron_B_eigpair
            
    
        lmd_diff.append((A_val*B_val-A_kron_B_val)/A_kron_B_val)
        subspace_angle.append(1-abs(np.dot(np.kron(B_vec,A_vec),A_kron_B_vec)))

        print(f"{file}: diff:{lmd_diff[-1]}   angle:{subspace_angle[-1]}   ARST_codes:{AReigST_codes[i]}")


    #
    #   Plotting Subroutines
    #
 


    def make_violin_plot(ax,data,precision=2,c=None,v_alpha=.3,format="default",xlim=None,xscale="linear",column_type=None):

        if xscale=="linear":
            v = ax.violinplot(data,[.5], points=100, showmeans=False,widths=.15,
                        showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))      
        elif xscale=="log":
            v = ax.violinplot(np.log10(data),[.5], points=100, showmeans=False,widths=.15,showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))  
        else:
            raise ValueError(f"supports xscale values: 'linear' & 'log'; Got {xscale}.")


        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        newMedianLines = [[(x0,y1-.01),(x0,y1+.7)]]
        v["cmedians"].set_segments(newMedianLines)

        # -- place extremal markers underneath
        extremal_tick_ypos = .25

        # -- write data values as text
        bbox = dict(boxstyle="round", ec="w", fc="w", alpha=.7,pad=.01)
        if column_type is None:

            if format == "default":
                ax.annotate(f"{np.median(data):.{precision}f}",xy=(.5,.375),xycoords="axes fraction",ha="center",fontsize=10).set_bbox(bbox)
                ax.annotate(f"{np.min(data):.{precision}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8).set_bbox(bbox)
                ax.annotate(f"{np.max(data):.{precision}f}",xy=(.975,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8).set_bbox(bbox)
            elif format == "scientific":
                ax.annotate(f"{np.median(data):.{precision}e}",xy=(.5,.375),xycoords="axes fraction",ha="center",fontsize=10).set_bbox(bbox)
                ax.annotate(f"{np.min(data):.{precision}e}",xy=(.025,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8).set_bbox(bbox)
                ax.annotate(f"{np.max(data):.{precision}e}",xy=(.975,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8).set_bbox(bbox)
            else:
                print(f"expecting format to be either 'default' or 'scientific', got:{format}")
        elif column_type == "merged_axis":
            pass
        else:
            raise ValueError("column_type expecting 'merged_axis' or None, but got {column_type}\n")

        if c is not None:
            v["cmedians"].set_color(c)
            v["cmedians"].set_alpha(.75)

                        

            for b in v['bodies']:
                # set colors 
  
                b.set_facecolor(c)
                b.set_edgecolor("None")
                b.set_alpha(.3)
   
                #  -- only plot the top half of violin plot --  #
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)
                
                #  -- clip the top of the med-line to the top of the violin plot --  #
                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )

                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])

 
    def make_violin_plot_legend(ax,c="k"):
        
        np.random.seed(12)#Ok looking:12
        data = [np.random.normal() for i in range(50)]
        v = ax.violinplot(data, points=100, positions=[.6], showmeans=False, 
                        showextrema=False, showmedians=True,widths=.6,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #ax.axes.set_axis_off()
        ax.set_ylim(.5,1.0)
        ax.patch.set_alpha(0.0) 
                # turn off background
        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        newMedianLines = [[(x0,y1 +.05),(x0,1.5)]]
        v["cmedians"].set_segments(newMedianLines)

        ax.annotate("median",xy=(.5,.325),xycoords="axes fraction",ha="center",va="center",fontsize=10)
        ax.annotate(f"min",xy=(.025,-.075),xycoords="axes fraction",ha="left",fontsize=9,alpha=.8)
        ax.annotate(f"max",xy=(.975,-.075),xycoords="axes fraction",ha="right",fontsize=9,alpha=.8)

        if c is not None:
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                
                b.set_facecolor(c)
                b.set_edgecolor("None")
                b.set_alpha(.3)
                
                #b.set_color(c)
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)

                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    new_max_y += .02
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])

    fig = plt.figure(figsize=(5.75,1))
    n = 1
    m = 2 # 4
    gs = fig.add_gridspec(1, m,
                        left=0.2, right=0.975,top=.975,bottom=.05,
                        wspace=.025,hspace=.1)

    all_axes = np.empty(m,object)
    for j in range(m):
        all_axes[j] = fig.add_subplot(gs[j])
                
    replace_zeros_with_machine_epsilon = lambda vals: [2e-16 if x == 0 else abs(x) for x in vals]

    make_violin_plot(all_axes[0],replace_zeros_with_machine_epsilon(lmd_diff),xscale="log",c="k",v_alpha=.3,format="scientific",precision=3)
    make_violin_plot(all_axes[1],replace_zeros_with_machine_epsilon(subspace_angle),xscale="log",c="k",v_alpha=.3,format="scientific",precision=3)
    
    #
    #   Touch up Axes
    #

    legend_axis = all_axes[0].inset_axes([-0.45,.3,.4,.5])
    make_violin_plot_legend(legend_axis)
    
    for ax in chain(all_axes.reshape(-1),[legend_axis]):
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis="both",which='major', length=0,pad=6)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    #for ax in all_axes:
        #ax.set_ylim(.3,.8)
    n = len(all_axes.reshape(-1))
    for (i,ax) in zip(range(n-1,0,-1),all_axes.reshape(-1)):
        ax.set_zorder(i)

    all_axes[0].annotate(r"$||\lambda_B||\lambda_A| - |\lambda_{B \otimes A}||/|\lambda_{B \otimes A}|$",xy=(0.5 , 0.2), xycoords='axes fraction',ha="center",va="top",fontsize=12)
    all_axes[1].annotate(r"$1-|\langle {\bf v_{B}} \otimes {\bf v_{A}},{\bf v_{B \otimes A}}\rangle|$",xy=(0.5 , 0.2), xycoords='axes fraction',ha="center",va="top",fontsize=12)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



#
#   TAME Low Rank Structure + Scaling
#
def TAME_vs_LRTAME_rank_1_case_singular_values(save_path=None):

    f = plt.figure(figsize=(7,3))
    gs = f.add_gridspec(nrows=1, ncols=3, left=0.05, right=0.975,wspace=0.3,hspace=0.1,top=.975,bottom=.175)
    all_ax = np.empty(3,object)
    for i in range(3):
        all_ax[i] = f.add_subplot(gs[i])

    TAME_LVGNA_rank_1_case_singular_values(all_ax[0])
    TAME_RandomGeometric_rank_1_singular_values_v2(all_ax[1],noiseModel="dupNoise")
    TAME_RandomGeometric_rank_1_singular_values_v2(all_ax[2],noiseModel="ERNoise")

    #
    #   Tweak Axis Details
    #

    all_ax[0].set_ylabel(r"$\sigma_2$",fontsize=14,labelpad=-5)

    for ax in all_ax[:2]:
        ax.annotate(r"$\epsilon$", xy=(1.125, .05), xycoords='axes fraction', c=purple_c,fontsize=12)

    for ax in all_ax:
        
        ax.set_ylim(5e-17,1e-7)
        ax.set_yticks([1e-16,1e-14,1e-12,1e-10,1e-08])
        ax.tick_params(axis="both",which='major', length=0,pad=6)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
    all_ax[0].set_yticklabels([])
    all_ax[0].set_xlim(2e5,4e11)
    for ax in all_ax[1:]:
        ax.set_xlim(3e5,3e11)
        ax.set_yticklabels(["",1e-14,1e-12,1e-10,1e-8])
        ax.set_xticks([1e6,1e7,1e8,1e9,1e10,1e11])
    
    title_size = 12
    x_loc = .125

    all_ax[0].annotate("LVGNA",xy=(x_loc,.875), xycoords='axes fraction',ha="left",va="top",fontsize=title_size)
    all_ax[1].annotate("Duplication\nNoise",xy=(x_loc,.875), xycoords='axes fraction',ha="left",va="top",fontsize=title_size)
    all_ax[2].annotate(u"Erdős Rényi"+"\nNoise",xy=(x_loc,.875), xycoords='axes fraction',ha="left",va="top",fontsize=title_size)

    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


#  - LRTAME has more accurate singular values for Triangle Adjancency tensors -  #
def TAME_LVGNA_rank_1_case_singular_values(ax=None):
    datapath = TAME_RESULTS + "rank1_singular_values/"

    with open(datapath + "TAME_LVGNA_iter_30_no_match_tol_1e-12.json","r") as f:
        TAME_data = json.load(f)
        TAME_data= TAME_data[-1]

    with open(datapath + "LowRankTAME_LVGNA_iter_30_no_match_tol_1e-12.json","r") as f:
        LowRankTAME_data = json.load(f)    
        LowRankTAME_data = LowRankTAME_data[-1]

    showFig = False
    if ax is None:
        f = plt.figure(figsize=(3,3))
        ax = plt.gca()
        showFig = True

    def process_data(data):
        nonzero_second_largest_sing_vals = []
        zero_second_largest_sing_vals = []

        nonzero_vertex_products = []
        zero_vertex_products = []

        nonzero_triangle_products = []   
        zero_triangle_products = []

        for file_A,file_B,_,_,profile in data:
            graph_A =  " ".join(file_A.split(".ssten")[0].split("_"))
            graph_B =  " ".join(file_B.split(".ssten")[0].split("_"))
            profile_dict = profile[0][-1]

            #normalize the singular values 
            for s in profile_dict["sing_vals"]:
                total = sum(s)
                s[:] = [s_i/total for s_i in s]

            #max_rank = int(max(profile_dict["ranks"]))
            #sing_vals = [s[1] if len(s) > 1 else 2e-16 for s in profile_dict["sing_vals"]
            sing_vals = [(i,s[1]) if len(s) > 1 else (i,2e-16) for (i,s) in enumerate(profile_dict["sing_vals"])]
            #find max sing val, and check iterates rank
            i,sing2_val = max(sing_vals,key=lambda x:x[1])
            rank = profile_dict["ranks"][i]

            if rank > 1:
            #if max_rank > 1.0:
                nonzero_second_largest_sing_vals.append(sing2_val)
                nonzero_vertex_products.append(vertex_counts[graph_A]*vertex_counts[graph_B])
                nonzero_triangle_products.append(triangle_counts[graph_A]*triangle_counts[graph_B])
            else:
                zero_second_largest_sing_vals.append(sing2_val)
                zero_vertex_products.append(vertex_counts[graph_A]*vertex_counts[graph_B])
                zero_triangle_products.append(triangle_counts[graph_A]*triangle_counts[graph_B])

        return nonzero_second_largest_sing_vals, nonzero_vertex_products, nonzero_triangle_products, zero_second_largest_sing_vals, zero_vertex_products, zero_triangle_products

    
    TAME_nonzero_second_largest_sing_vals, TAME_nonzero_vertex_products, \
        TAME_nonzero_triangle_products, TAME_zero_second_largest_sing_vals,\
             TAME_zero_vertex_products, TAME_zero_triangle_products = process_data(TAME_data)


    
    LowRankTAME_nonzero_second_largest_sing_vals, LowRankTAME_nonzero_vertex_products, \
        LowRankTAME_nonzero_triangle_products, LowRankTAME_zero_second_largest_sing_vals,\
             LowRankTAME_zero_vertex_products, LowRankTAME_zero_triangle_products = process_data(LowRankTAME_data)


    ax.set_yscale("log")
    ax.grid(which="major", axis="both")

    if showFig:
        ax.annotate(r"$\epsilon$", xy=(1.06, .02), xycoords='axes fraction', c=purple_c)
        ax.set_ylabel(r"$\sigma_2$")
        ax.set_ylim(1e-16,1e-7)

    
          

    ax.annotate("TAME", xy=(.6,.55),xycoords='axes fraction',fontsize=12,c=T_color)
    ax.annotate("LowRankTAME", xy=(.2,.1),xycoords='axes fraction',fontsize=12,c=LRT_color)
  
    ax.set_xlabel(r"|$T_A$||$T_B$|")
    ax.set_yscale("log")
    ax.set_xscale("log")


    #scatter plot formatting
    marker_size = 25
    marker_alpha = .5

    ax.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o',c=T_color,s=marker_size,alpha=marker_alpha)
    ax.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=T_color,s=marker_size)

    #plot machine epsilon
    ax.plot([1e5,1e13],[2e-16]*2,c=purple_c,zorder=1)

    ax.scatter(LowRankTAME_nonzero_triangle_products,LowRankTAME_nonzero_second_largest_sing_vals,marker='o', c=LRT_color,s=marker_size,alpha=marker_alpha)
    scatter = ax.scatter(LowRankTAME_zero_triangle_products,LowRankTAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=LRT_color,s=marker_size,zorder=2)
    
    ax.set_xlim(7e4,1e12)
    ax.set_xticks([1e6,1e7,1e8,1e9,1e10,1e11])  
    if showFig:
        plt.show()
    
#Shows the noise introduced by the TAME routine by considering second largest 
# singular values in the rank 1 case (alpha=1.0, beta =0.0), plots againts both 
# |V_A||V_B| and |T_A||T_B| for comparison. Data plotted for RandomGeometric Graphs 
# degreedist = LogNormal(5,1). 
def TAME_RandomGeometric_rank_1_case_singular_values(axes=None):
    """
        Note: This an old figure 
    """

    if axes is None:
        f,axes = plt.subplots(1,1,dpi=60)
        f.set_size_inches(3, 3)

   
    with open(TAME_RESULTS + "Rank1SingularValues/LowRankTAME_RandomGeometric_log5_iter_30_n_100_20K_no_match_tol_1e-12.json","r") as f:
        LowRankTAME_data = json.load(f)    

    with open(TAME_RESULTS + "Rank1SingularValues/TAME_RandomGeometric_degreedist:log5_alphas:[1.0]_betas:[0.0]_iter:30_trials:10_n:[1e2,5e2,1e3,2e3,5e3,1e4,2e4]_no_match_tol:1e-12.json","r") as f:
        TAME_data = json.load(f)
    

    def process_RandomGeometricResults(data):

        nonzero_second_largest_sing_vals = []
        zero_second_largest_sing_vals = []

        nonzero_vertex_products = []
        zero_vertex_products = []

        nonzero_triangle_products = []   
        zero_triangle_products = []

        n_values = set()
        for p,seed,p_remove,n,_,max_tris,profiles in data:
            n_values.add(n)
            params, profile_dict = profiles[0] #should only be alpha = 1.0, beta = 0.0

            #normalize the singular values
            for s in profile_dict["sing_vals"]: 
                s = [0.0 if x is None else x for x in s]
                #saving to json seems to introduce Nones when reading from saved Julia files
                total = sum(s)
                s[:] = [s_i/total for s_i in s]

            #max_rank = int(max(profile_dict["ranks"]))

            sing_vals = [(i,s[1]) if len(s) > 1 else (i,2e-16) for (i,s) in enumerate(profile_dict["sing_vals"])]
            #find max sing val, and check iterates rank
            i,sing2_val = max(sing_vals,key=lambda x:x[1])
            rank = profile_dict["ranks"][i]

            if rank > 1:
                #print([sum(s) for s in profile_dict["sing_vals"]])
                nonzero_second_largest_sing_vals.append(sing2_val)
                nonzero_vertex_products.append(n**2)
                nonzero_triangle_products.append(max_tris**2) 

                #TODO: need to use seed to compute actual triangle counts
            else:
                zero_second_largest_sing_vals.append(sing2_val)
                zero_vertex_products.append(n**2)
                zero_triangle_products.append(max_tris**2) 

        return n_values, nonzero_second_largest_sing_vals, nonzero_vertex_products, nonzero_triangle_products, zero_second_largest_sing_vals, zero_vertex_products, zero_triangle_products


    n_values, LowRankTAME_nonzero_second_largest_sing_vals, LowRankTAME_nonzero_vertex_products,\
         LowRankTAME_nonzero_triangle_products, LowRankTAME_zero_second_largest_sing_vals,\
              LowRankTAME_zero_vertex_products, LowRankTAME_zero_triangle_products =\
                   process_RandomGeometricResults(LowRankTAME_data)
    _, TAME_nonzero_second_largest_sing_vals, TAME_nonzero_vertex_products,\
         TAME_nonzero_triangle_products, TAME_zero_second_largest_sing_vals,\
              TAME_zero_vertex_products, TAME_zero_triangle_products =\
                   process_RandomGeometricResults(TAME_data)

    
    #
    #   Make Triangle_Triangle plots
    #
    ax = axes
    #ax = plt.subplot(122)

    #format the axis
    ax.set_yscale("log")
    ax.grid(which="major", axis="y")
    #ax.set_ylabel(r"max $\sigma_2")
    #ax.set_ylabel(r"max [$\sum_{i=2}^k\sigma_i]")
    #ax.yaxis.set_ticks_position('right')
    #ax.tick_params(labeltop=False, labelright=True)
    ax.set_xlabel(r"|$T_A$||$T_B$|")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(3e5,7e11)
    ax.set_xticks([1e7,1e8,1e11])
    #ax.set_ylim(1e-16,1e-7)
    ax.annotate(r"$\epsilon$", xy=(1.06, .02), xycoords='axes fraction', c=purple_c)
    ax.set_ylabel(r"$\sigma_2$")
    #scatter plot formatting
    marker_size = 20
    marker_alpha = 1.0

    #plot the TAME Data
    """
    plt.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o',c=darker_t4_color,s=marker_size)
    plt.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=darker_t4_color,s=marker_size)
    """
    #plot machine epsilon
    plt.plot([3e5,7e11],[2e-16]*2,c=purple_c,zorder=1)

    #plot LowRankTAME Data
    plt.scatter(LowRankTAME_nonzero_triangle_products,LowRankTAME_nonzero_second_largest_sing_vals,marker='o', c=LRT_color,s=marker_size,zorder=2)
    plt.scatter(LowRankTAME_zero_triangle_products,LowRankTAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=LRT_color,s=marker_size,zorder=2)

    #plot TAME Data
    plt.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o', c=T_color,s=marker_size,zorder=3)
    plt.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=T_color,s=marker_size,zorder=3)


    axins = ax.inset_axes([.6,.15,.25,.25]) # zoom = 6
    axins.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o', c=T_color,s=marker_size)
    axins.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=T_color,s=marker_size)
    # sub region of the original image
    #axins.set_xlim(9e9, 3e11)
    axins.set_xlim(4e9, 1e10)
    axins.set_ylim(5e-13, 1.5e-12)
    axins.set_xscale("log")
    axins.set_yscale("log")
    axins.set_xticks([])
    axins.minorticks_off()
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5",alpha=.5,zorder=1)

   
    #axins.tick_params(labelleft=False, labelbottom=False)
    #axins.set_yticks([])

    """
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    """


    #TODO: combining plots, potentially remove later
    plt.tight_layout()
    plt.show()

def TAME_data_extraction():
    results_path = TAME_RESULTS + "MaxRankExperiments/"
    T_file = "TAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0,10.0,100.0]_n:[100,500,1000,2000,5000,10000]_p:[0.5]_noiseModel:Duplication_sp:[0.25]_trials:50_MaxRankResults.json"
    
    new_data = []
    new_file = "TAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:[1.0]_beta:[0.0]_n:[100,500,1000,2000,5000,10000]_p:[0.5]_noiseModel:Duplication_sp:[0.25]_trials:50_MaxRankResults.json"
    
    with open(results_path+T_file,"r") as f:
        TAME_data = json.load(f)

        for (seed,p,n,sp,acc,dw_acc,tri_match,A_tri,B_tri,max_tris,profiles) in TAME_data:
            new_profiles = []
            for (param, profiling) in profiles:
                if param == "α:1.0_β:0.0":
                    new_profiles.append([param, profiling])

            new_data.append((seed,p,n,sp,acc,dw_acc,tri_match,A_tri,B_tri,max_tris,new_profiles))

    with open(results_path+new_file,"w") as f:
         json.dump(new_data,f)
    return new_data

def TAME_RandomGeometric_rank_1_singular_values_v2(ax=None,noiseModel="dupNoise"):

    showFig=False
    if ax is None:
        f = plt.figure(dpi=60)
        ax = plt.gca()
        f.set_size_inches(3, 4)
        showFig=True

    #
    #    Load the Data
    #

    results_path = TAME_RESULTS + "MaxRankExperiments/"
    results_path  = TAME_RESULTS + "TAME_iterate_max_rank/"

    def process_data(data):
        nonzero_second_largest_sing_vals = []
        zero_second_largest_sing_vals = []

        nonzero_vertex_products = []
        zero_vertex_products = []

        nonzero_triangle_products = []   
        zero_triangle_products = []

        n_values = set()

        for datum in data:
            
            if noiseModel == "dupNoise":
                (seed,p,n,sp,acc,dw_acc,tri_match,A_tri,B_tri,max_tris,profiles) = datum
            elif noiseModel == "ERNoise":
                (seed,p,n,acc,dw_acc,tri_match,A_tri,B_tri,max_tris,profiles) = datum

            for (params, profile_dict) in profiles:
                if params == "α:1.0_β:0.0":
                    #normalize the singular values
                    for s in profile_dict["sing_vals"]: 
                        s = [0.0 if x is None else x for x in s]
                        #saving to json seems to introduce Nones when reading from saved Julia files
                        total = sum(s)
                        s[:] = [s_i/total for s_i in s]

                    sing_vals = [(i,s[1]) if len(s) > 1 else (i,2e-16) for (i,s) in enumerate(profile_dict["sing_vals"])]
                    #find max sing val, and check iterates rank
                    i,sing2_val = max(sing_vals,key=lambda x:x[1])
                    rank = profile_dict["ranks"][i]

                    if rank > 1:
                        #print([sum(s) for s in profile_dict["sing_vals"]])
                        nonzero_second_largest_sing_vals.append(sing2_val)
                        nonzero_vertex_products.append(n**2)
                        nonzero_triangle_products.append(max_tris**2) 

                        #TODO: need to use seed to compute actual triangle counts
                    else:
                        zero_second_largest_sing_vals.append(sing2_val)
                        zero_vertex_products.append(n**2)
                        zero_triangle_products.append(max_tris**2) 

        return nonzero_second_largest_sing_vals, zero_second_largest_sing_vals,\
               nonzero_vertex_products, zero_vertex_products,\
               nonzero_triangle_products, zero_triangle_products,\
               n_values

    if noiseModel == "dupNoise":
        LRT_file = "LRTAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0,10.0,100.0]_n:[100,500,1000,2000,5000,10000]_p:[0.5]_noiseModel:Duplication_sp:[0.25]_trials:50_MaxRankResults.json"
    elif noiseModel == "ERNoise":
        LRT_file = "LRTAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0,10.0,100.0]_n:[100,500,1000,2000,5000,10000]_p:[0.05]_noiseModel:ER_trials:50_MaxRankResults.json"
    
    
    with open(results_path + LRT_file,"r") as f:
    
        LowRankTAME_nonzero_second_largest_sing_vals, LowRankTAME_zero_second_largest_sing_vals,\
        LowRankTAME_nonzero_vertex_products,   LowRankTAME_zero_vertex_products,\
        LowRankTAME_nonzero_triangle_products, LowRankTAME_zero_triangle_products,\
        n_values = process_data(json.load(f))



    #TODO: extra out the \alpha:1.0, \beta = 0.0 case into a seperate file
    #T_file = "TAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0,10.0,100.0]_n:[100,500,1000,2000,5000,10000]_p:[0.5]_noiseModel:Duplication_sp:[0.25]_trials:50_MaxRankResults.json"
    if noiseModel == "dupNoise":
        T_file = "TAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:[1.0]_beta:[0.0]_n:[100,500,1000,2000,5000,10000]_p:[0.5]_noiseModel:Duplication_sp:[0.25]_trials:50_MaxRankResults.json"
    elif noiseModel == "ERNoise":
        T_file = "TAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:[1.0]_beta:[0.0]_n:[100,500,1000,2000,5000,10000]_p:[0.05]_noiseModel:ER_trials:50_MaxRankResults.json"
    
    with open(results_path+T_file,"r") as f:
        TAME_nonzero_second_largest_sing_vals, TAME_zero_second_largest_sing_vals,\
        TAME_nonzero_vertex_products,   TAME_zero_vertex_products,\
        TAME_nonzero_triangle_products, TAME_zero_triangle_products,\
        n_values = process_data(json.load(f))



    #
    #   Plot the Data
    #
    ax.set_yscale("log")
    ax.grid(which="major", axis="both")
    #ax.set_ylabel(r"max $\sigma_2")
    #ax.set_ylabel(r"max [$\sum_{i=2}^k\sigma_i]")
    #ax.yaxis.set_ticks_position('right')
    #ax.tick_params(labeltop=False, labelright=True)
    ax.set_xlabel(r"|$T_A$||$T_B$|")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(3e5,3e11)
    ax.set_xticks([1e7,1e8,1e11])
    #ax.set_ylim(1e-16,1e-7)
    if showFig:
        ax.annotate(r"$\epsilon$", xy=(1.06, .02), xycoords='axes fraction', c=purple_c)
        ax.set_ylabel(r"$\sigma_2$")
    #scatter plot formatting
    marker_size = 25
    marker_alpha = .2

    #plot the TAME Data
    """
    plt.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o',c=darker_t4_color,s=marker_size)
    plt.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=darker_t4_color,s=marker_size)
    """
    #plot machine epsilon
    ax.plot([3e5,7e11],[2e-16]*2,c=purple_c,zorder=1)

    #plot LowRankTAME Data
    ax.scatter(LowRankTAME_nonzero_triangle_products,LowRankTAME_nonzero_second_largest_sing_vals,marker='o', c=LRT_color,s=marker_size,zorder=2,alpha=marker_alpha)
    ax.scatter(LowRankTAME_zero_triangle_products,LowRankTAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=LRT_color,s=marker_size,zorder=2)

    #plot TAME Data
    ax.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker="o", c=T_color,s=marker_size,zorder=3,alpha=marker_alpha)
    ax.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,marker="o",facecolors='none',edgecolors=T_color,s=marker_size,zorder=3)

    #
    #   Add in inset axes
    #
    """
    axin_axes = []
    

    if noiseModel == "dupNoise":
        box_parameters = [
            ([.4,.125,.125,.2],(9e7, 4.5e8),(1e-13, 8e-13)),
            ([.525,.125,.125,.2],(7e8, 2e9),(3e-13, 1e-12)),
            ([.65,.125,.125,.2],(4e9, 8.5e9),(5e-13, 1.5e-12)),
            ([.775,.125,.125,.2],(2e10, 4e10),(6e-13, 3e-12)),   
        ]
    else:
        box_parameters = [
            ([.4,.125,.125,.2],(2e8, 2e9),(5e-13, 1e-12)),
            ([.525,.125,.125,.2],(8e8, 2e9),(3e-13, 1e-12)),
            ([.65,.125,.125,.2],(6e9, 1e10),(8e-13, 2e-12)),
            ([.775,.125,.125,.2],(2e10, 5e10),(9.5e-13, 3.5e-12)),   
        ]

    for (axin_loc,xlims,ylims) in box_parameters:

        axins = ax.inset_axes(axin_loc) # zoom = 6
        axin_axes.append(axins)

        axins.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o', c=T_color,s=marker_size,alpha=marker_alpha)
        axins.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=T_color,s=marker_size)
        # sub region of the original image
        #axins.set_xlim(9e9, 3e11)
        axins.set_xlim(*xlims)
        axins.set_ylim(*ylims)

        axins.set_xscale("log")
        axins.set_yscale("log")
        axins.set_xticks([])
        axins.minorticks_off()
        axins.set_yticks([])
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5",alpha=.5,zorder=1)
    """


    """
    if noiseModel == "dupNoise":
        axins = ax.inset_axes([.3,.1,.4,.2])
    else:
        axins = ax.inset_axes([.6,.15,.25,.25])
    
    axins.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o', c=T_color,s=marker_size)
    axins.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=T_color,s=marker_size)
    # sub region of the original image
    if noiseModel == "dupNoise":
        #axins.set_xlim(9e9, 3e11)
        axins.set_xlim(5e7, 1e10)
        axins.set_ylim(1e-13, 1e-12)
    else:
        axins.set_xlim(4e9, 1e10)
        axins.set_ylim(5e-13, 1.5e-12)

    axins.set_xscale("log")
    axins.set_yscale("log")
    axins.set_xticks([])
    axins.minorticks_off()
    axins.set_yticks([])

    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5",alpha=.5,zorder=1)
    """

    if showFig:
        plt.tight_layout()
        plt.show()


#  - TAME iterates are low rank -  #

def max_rank_experiments(save_path=None):

    #f = plt.figure(dpi=60)
    #f.set_size_inches(10, 4)
    f = plt.figure(figsize=(5.2,5))
    n = 3 
    m = 2 
    gs = f.add_gridspec(nrows=n, ncols=m, left=0.05, right=0.975,wspace=0.225,hspace=0.125,top=.975,bottom=.1)
    all_ax = np.empty((n,m),object)
    for i in range(n):
        for j in range(m):
            if j == 0:
                all_ax[i,j] = f.add_subplot(gs[i,j])
            else:
                all_ax[i,j] = f.add_subplot(gs[i,j],sharex=all_ax[i,0])


    DupNoise_data = "LRTAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0,10.0,100.0]_n:[100,500,1000,2000,5000,10000]_p:[0.5]_noiseModel:Duplication_sp:[0.25]_trials:50_MaxRankResults.json"
    ERNoise_data = "LRTAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0,10.0,100.0]_n:[100,500,1000,2000,5000,10000]_p:[0.05]_noiseModel:ER_trials:50_MaxRankResults.json"
    
    max_rank_synthetic_data(all_ax[0,:],ERNoise_data,"ER")
    max_rank_synthetic_data(all_ax[1,:],DupNoise_data,"Duplication")
    max_rank_LVGNA_data(all_ax[2,:])
    
    checkboard_color = [.925]*3

    for ax in all_ax.reshape(-1):
        ax.grid(which="major", axis="both")
        ax.set_xscale("log")
        ax.set_xlim(1e5,1e12)

    #
    # -- handle tick marks -- #
    #
    for i,ax in enumerate(all_ax[:,1]):
        ax.set_ylabel("max rank\n min{n,m}",ha="center")
        #ax.set_ylabel(r"max rank/$\min{\{n,m\}}$")
        ax.annotate('', xy=(-.1, .25), xycoords='axes fraction', xytext=(-.1, 0.75),
                        arrowprops=dict(arrowstyle="-", color='k'))
        
        if i != 2:
            tick_ha = "right"
        else:
            tick_ha = "left"
        
        for tick in ax.yaxis.get_majorticklabels():
                tick.set_horizontalalignment(tick_ha)
        ax.tick_params(axis="y",which="both",direction="in",pad=-25)
        ax.set_yscale("log")
        if i != 2:
            ax.set_ylim(5e-3,1.5)
        else:
            ax.set_ylim(5e-4,2e-1)

        # -- add in Plot labels
        x_loc = .85
        title_size = 12
        if i == 0:
            ax.annotate(u"Erdős Rényi"+"\nNoise",xy=(.9,.675), xycoords='axes fraction',ha="right",fontsize=title_size)
        elif i == 1:
            ax.annotate("Duplication\n Noise",xy=(.9,.675), xycoords='axes fraction',ha="right",fontsize=title_size)
        elif i == 2:
            x_loc = .85
            ax.annotate("LVGNA",xy=(x_loc,.7), xycoords='axes fraction',ha="right",fontsize=title_size)

        
        #ax.yaxis.set_ticks_position('right')
        if i != 2:
            ax.tick_params(axis="x",direction="out",which='both', length=0)

    #ax.set_ylabel("maximum rank")
    all_ax[2,1].tick_params(labeltop=False, labelright=True)

    #  -- set labels --  #
    for i,ax in enumerate(all_ax[:,0]):
        ax.set_ylabel("max rank")
        ax.set_xlim(5e4,5e11)
        ax.set_xticks([1e5,1e6,1e7,1e8,1e9,1e10,1e11])
        ax.tick_params(axis="y",which="both",direction="in",pad=-25)
        if i != 2:
            ax.set_ylim(0,320)
            ax.tick_params(axis="x",direction="out",which='both', length=0)
            ax.set_yticks([50,100,150,200,250,300])
            ax.set_xticklabels([])
            #for tl in ax.get_yticklabels():
            #        tl.set_bbox(bbox)
        else:
            ax.set_ylim(0,150)
            ax.set_yticks([50,100,125])
            ax.set_xticklabels([1e5,1e6,1e7,1e8,1e9,1e10,1e11])
        
        # -- add in annotations  

    all_ax[2,0].xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())

    for j in range(m):
        #parity = 1
        for (i,ax) in enumerate(all_ax[:,j]):
            parity = 1
            for pos in ['left','bottom','top','right']:
                ax.spines[pos].set_visible(False)
            if j == 1:
                ax.tick_params(axis="y",which='minor', length=0)

            if i == 2:
                ax.yaxis.set_ticks_position('right')
            """
            if i == 0:
                ax.spines['bottom'].set_visible(False)
            elif i == 2:
                ax.spines['top'].set_visible(False)
                ax.yaxis.set_ticks_position('right')
            else:
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
            """
            if parity == -1:
                all_ax[i,j].patch.set_facecolor(checkboard_color)
            
            if parity == 1:
                bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.1)
            else:
                bbox = dict(boxstyle="round", ec=checkboard_color, fc=checkboard_color, alpha=1.0,pad=.1)
   
            for tl in ax.get_yticklabels(): 
                tl.set_bbox(bbox)
            parity *= -1
        parity *= -1


            
    for ax in all_ax[2,:]:
        ax.set_xlabel(r"$|T_A||T_B|$")
        ax.tick_params(axis="x",direction="out",which='both', length=0)


    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def max_rank_LVGNA_data(axes,includeSampledEigShift=False):

    data_path = TAME_RESULTS + "TAME_iterate_max_rank/"
    with open(data_path + "LowRankTAME_LVGNA.json", 'r') as f:
        MM_results  = json.load(f)
    MM_exp_data, param_indices,MM_file_names = process_TAME_output2(MM_results[1])

    if includeSampledEigShift:
        beta_100_loc = (.475, .75)
    else:
        beta_100_loc = (.05, .10)

    label_meta = [
        ((.05, .35),"o",red_c,"solid"),
        ((.05, .575),"v",green_shift_c,"dotted"),
        ((.075, .75),"*",purple_c,"dashed"),
        (beta_100_loc,"s",blue_c,"dashdot")
    ]


    for (param, j),(loc,marker,c,linestyle) in zip(param_indices.items(),label_meta):
        n_points = []
        max_ranks = []
        normalized_max_ranks = []
        param_label = f"β:{int(float(param.split('β:')[-1]))}"
        axes[0].annotate(param_label, xy=loc, xycoords='axes fraction',c=c)

        for i in range(MM_exp_data.shape[0]):

            graph_names = [str.join(" ", x.split(".ssten")[0].split("_")) for x in MM_file_names[i]]
            min_n = np.min([vertex_counts[f] for f in graph_names])
            tri_counts = triangle_counts[graph_names[0]]*triangle_counts[graph_names[1]]
           # tri_counts = sum([triangle_counts[graph_names[0]] for f in graph_names])
            n_points.append(tri_counts)
            max_ranks.append(np.max(MM_exp_data[i,:,j,:]))
            normalized_max_ranks.append(np.max(MM_exp_data[i,:,j,:])/min_n)

        plot_1d_loess_smoothing(n_points,max_ranks,.3,axes[0],c=c)#,linestyle=linestyle
        plot_1d_loess_smoothing(n_points,normalized_max_ranks,.3,axes[1],c=c,logFilter=True) #linestyle=linestyle,
       

    if includeSampledEigShift:
        (filename,color,label,loc,linestyle) = (
            "LVGNAMaxEigShiftRanks_alphas:[0.5,1.0]_iter:30_SSHOPMSamples:1000_tol:1.0e-16_results.json",
            [.25]*3,
            "β:"+r"$\lambda_A\lambda_B$",
            (-.05,.02),
            (0,(3,1,1,1,1,1)))
        
        with open(data_path + filename,"r") as f:
            data = json.load(f)

            tri_products = []
            max_ranks = []
            normalized_max_ranks = []

            tri_count =lambda f: triangle_counts[str.join(" ", f.split(".ssten")[0].split("_"))]
            for (graphA, graphB,shiftA,shiftB,profiles) in data:

                graph_names = [str.join(" ", x.split(".ssten")[0].split("_")) for x in [graphA, graphB]]
                min_n = np.min([vertex_counts[f] for f in graph_names])
                
                tri_products.append(tri_count(graphA)*tri_count(graphB))

                max_ranks.append(max([max(profile["ranks"]) for (p,profile) in profiles]))
                normalized_max_ranks.append(max([max(profile["ranks"]) for (p,profile) in profiles])/min_n)


            plot_1d_loess_smoothing(tri_products,max_ranks,.3,axes[0],c=color,logFilter=True) #,linestyle=linestyle
            plot_1d_loess_smoothing(tri_products,normalized_max_ranks,.3,axes[1],c=color,logFilter=True)#,linestyle=linestyle
            
            axes[0].annotate(label, xy=loc, xycoords='axes fraction',c=color)

def max_rank_synthetic_data(axes,filename,noise_model="ER"):
    #with open(TAME_RESULTS + "MaxRankExperiments/","r") as f:

    with open(TAME_RESULTS + "TAME_iterate_max_rank/" + filename,"r") as f:
    #with open(TAME_RESULTS + "MaxRankExperiments/LowRankTAME_RandomGeometric_degreedist_log5_iter:15_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[100,500,1K,2K,5K,10K,20K]_noMatching_pRemove:[.01,.05]_tol:1e-12_trialcount:50.json","r") as f:
        synth_results = json.load(f)
    
    MM_exp_data, n_vals, p_vals, param_vals, tri_counts = process_synthetic_TAME_output2(synth_results,noise_model)
 
    print(np.mean(MM_exp_data))

    #ax = axes[0]
    #ax.set_xscale("log") 
    #ax.set_ylim(00,315)

    if filename == "LRTAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0,10.0,100.0]_n:[100,500,1000,2000,5000,10000]_p:[0.05]_noiseModel:ER_trials:50_MaxRankResults.json":
        label_meta = [
            ((.825, .09),"o",red_c,"solid"),
            ((.825, .26),"v",green_shift_c,"dotted"),
            ((.825, .6), "*",purple_c,"dashed"),
            ((.825, .8), "s",blue_c,"dashdot")]
    else:
        label_meta = [
            ((.85, .11), "o",red_c,"solid"),
            ((.85, .25),"v",green_shift_c,"dotted"),
            ((.85, .55), "*",purple_c,"dashed"),
            ((.85, .825), "s",blue_c,"dashdot")]#  no_ylim_points

    #label_meta = [((.92, .25),t1_color),((.8, .51),t2_color),((.625, .8),purple_c),((.25, .88),t4_color)]

    #beta=100 annotations
    beta_100_meta = [(.15,.4),(.42,1.01),(.51,1.01),(.6,1.01),(.725,1.01),(.8,1.01),(.9,1.01)]

    #beta=10 annotations
    beta_10_meta = [(.15,.32),(.42,.515),(.51,.6),(.6,.65),(.725,.75),(.8,.95),(.9,.95)]

    for ((params,k),(loc,marker,c,linestyle)) in zip(param_vals.items(),label_meta):
        max_vals = []
        normalized_max_vals = []
        mean_tris = []
        
        for n,j in n_vals.items():
            max_vals.append(np.max(MM_exp_data[:,j,:,k,:]))
            normalized_max_vals.append(np.max(MM_exp_data[:,j,:,k,:])/n)
            mean_tris.append(np.mean(tri_counts[:,j,:,k]))
        

        axes[0].plot(mean_tris,max_vals,c=c)#,linestyle=linestyle
        axes[1].plot(mean_tris,normalized_max_vals,c=c)#,linestyle=linestyle

        param_label = f"β:{int(float(params.split('β:')[-1]))}"
        axes[0].annotate(param_label, xy=loc, xycoords='axes fraction',c=c)


#  - TAME & LRTAM don't scale for larger motifs -  #

def TAME_vs_LRTAME_clique_scaling_detailed(save_path=None,global_ax= None):
    plt.rc('text.latex', preamble=r'\usepackage{/Users/charlie/Documents/Code/TKPExperimentPlots/latex/dgleich-math}\usepackage{amsmath}')

    #
    #  subplot routines 
    #


    def make_violin_plot(ax,data,precision=2,c=None,v_alpha=.3,format="default",xlim=None,xscale="linear",column_type=None):
       
        if xscale=="linear":
            v = ax.violinplot(data,[.5], points=100, showmeans=False,widths=.15,
                        showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        elif xscale=="log":
            v = ax.violinplot(np.log10(data), points=100, showmeans=False,widths=.15,showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        else:
            raise ValueError(f"supports xscale values: 'linear' & 'log'; Got {xscale}.")


        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]

        newMedianLines = [[(x0,y1),(x0,y1+.7)]]
        v["cmedians"].set_segments(newMedianLines)


        # -- write data values as text
        bbox = dict(boxstyle="round", ec="w", fc="w", alpha=.5,pad=.05)
        extremal_tick_ypos = .1
        if column_type is None:

            if format == "default":
                ax.annotate(f"{np.median(data):.{precision}f}",xy=(.5,.4),xycoords="axes fraction",ha="center",fontsize=10)#.set_bbox(bbox)
                ax.annotate(f"{np.min(data):.{precision}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8).set_bbox(bbox)
                ax.annotate(f"{np.max(data):.{precision}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8).set_bbox(bbox)
            elif format == "scientific":
                ax.annotate(f"{np.median(data):.{precision}e}",xy=(.5,.4),xycoords="axes fraction",ha="center",fontsize=10).set_bbox(bbox)
                ax.annotate(f"{np.min(data):.{precision}e}",xy=(.025,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8).set_bbox(bbox)
                ax.annotate(f"{np.max(data):.{precision}e}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8).set_bbox(bbox)
            else:
                print(f"expecting format to be either 'default' or 'scientific', got:{format}")
        elif column_type == "merged_axis":
            pass
        else:
            raise ValueError("column_type expecting 'merged_axis' or None, but got {column_type}\n")

        if c is not None:
            v["cmedians"].set_color(c)
            v["cmedians"].set_alpha(.75)
            for b in v['bodies']:
                # set colors 
                b.set_facecolor(c)
                b.set_edgecolor("None")            
                b.set_alpha(v_alpha)

                #  -- only plot the top half of violin plot --  #
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)
                
                #  -- clip the top of the med-line to the top of the violin plot --  #
                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )

                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])

 

    def make_violin_plot_legend(ax,c="k"):
        
        np.random.seed(12)#Ok looking:12
        v = ax.violinplot([np.random.normal() for i in range(50)], points=100, positions=[.6], showmeans=False, 
                        showextrema=False, showmedians=True,widths=.6,vert=False)#,quantiles=[[.2,.8]]*len(data2))

        ax.set_ylim(.5,1.0)
        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]

        newMedianLines = [[(x0,y1 +.05),(x0,1.5)]]
        v["cmedians"].set_segments(newMedianLines)

        ax.annotate("median",xy=(.5,.35),xycoords="axes fraction",ha="center",va="center",fontsize=10)
        ax.annotate(f"min",xy=(.025,-.125),xycoords="axes fraction",ha="left",fontsize=9,alpha=.8)
        ax.annotate(f"max",xy=(.975,-.125),xycoords="axes fraction",ha="right",fontsize=9,alpha=.8)

        if c is not None:
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor(c)
                b.set_alpha(.3)
                b.set_color(c)
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)

                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    new_max_y += .02
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])



    #
    #  Parse Data
    #

    results_path = TAME_RESULTS + "RG_DupNoise/"

    elemwise_list_sum =lambda l1,l2: [a + b for (a,b) in zip(l1,l2)]
    filename = "LowRankTAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7,8]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"

    with open(results_path + filename,"r") as file:
        data = json.load(file)
        LR_results = {}
        debug_results = {}
        LR_seeds = {}

        for (order, LRTAME_output) in data:
            trials = len(LRTAME_output)
            LR_results[order] = {
                "full runtime":{},
                "contraction runtime":{},
                "A motifs":[],
                "ranks":{},
            }
            debug_results[order] = {
                "runtimes":[],
                "ranks":[]
            }
            LR_seeds[order] = []
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profiling) in enumerate(LRTAME_output):
                LR_results[order]["A motifs"].append(A_motifs)
                LR_seeds[order].append(seed)
                for (params,profile) in profiling:

                    rt = reduce(elemwise_list_sum,[
                        profile["low_rank_factoring_timings"],
                        profile["contraction_timings"],
                        profile["matching_timings"],
                        profile["scoring_timings"],
                        ])
                    if params in LR_results[order]["full runtime"]:
                        LR_results[order]["full runtime"][params].append(rt)
                    else:
                        LR_results[order]["full runtime"][params] = [rt]

                    contract_rt = reduce(elemwise_list_sum,[
                        profile["low_rank_factoring_timings"],
                        profile["contraction_timings"],
                        ])
                    if params in LR_results[order]["contraction runtime"]:
                        LR_results[order]["contraction runtime"][params].append(contract_rt)
                    else:
                        LR_results[order]["contraction runtime"][params] = [contract_rt]


                    if params == 'α:0.5_β:1.0':
                        debug_results[order]["ranks"].append(profile["ranks"])
                        debug_results[order]["runtimes"].append(contract_rt)

                    if params in LR_results[order]["ranks"]:
                        LR_results[order]["ranks"][params].append(profile["ranks"])
                    else:
                        LR_results[order]["ranks"][params] = [profile["ranks"]]

            
            for key in ["contraction runtime","full runtime"]:
                for (param,rts) in LR_results[order][key].items():
                    LR_results[order][key][param] = np.median(np.array(rts),axis=0)
                    

    
    def get_TAME_data():
            
        file = "TAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[25]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"
        files = [
            "TAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json",
            "TAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[8]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json",
        ]
        data = []
        for file in files:
            with open(results_path + file,"r") as f:
                data.extend(json.load(f))

        T_results = {}

        for (order, TAME_output) in data:
            trials = len(TAME_output)
            T_results[order] = {
                "contraction runtime":{},
                "full runtime":{},
            }

            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profiling) in enumerate(TAME_output):
                for (params,profile) in profiling:
                    rt = reduce(elemwise_list_sum,[
                        profile["contraction_timings"],
                        profile["matching_timings"],
                        profile["scoring_timings"],
                        ])

                    if params in T_results[order]:
                        T_results[order]["full runtime"][params].append(rt)
                    else:
                        T_results[order]["full runtime"][params] = [rt]

                    if params in T_results[order]:
                        T_results[order]["contraction runtime"][params].append(profile["contraction_timings"])
                    else:
                        T_results[order]["contraction runtime"][params] = [profile["contraction_timings"]]

            for key in ["contraction runtime","full runtime"]:
                for (param,rts) in T_results[order][key].items():
                    T_results[order][key][param] = np.median(np.array(rts),axis=0)
            """
            for params in T_results[order]["contraction runtime"].keys():
                T_results[order]["contraction runtime"][params] = [rt/trials for rt in T_results[order]["contraction runtime"][params]]
            for params in T_results[order]["full runtime"].keys():
                T_results[order]["full runtime"][params] = [rt/trials for rt in T_results[order]["full runtime"][params]]
            """
        return T_results
    
    T_results = get_TAME_data()
    
    file = "LambdaTAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"
    with open(results_path + file,"r") as file:
        data = json.load(file)
        LT_results = {}

        all_shifts = [
            'α:0.5_β:0.0',
            'α:0.5_β:1.0',
            'α:1.0_β:0.0',
            'α:1.0_β:1.0',
        ]
        for (order, TAME_output) in data:
            trials = len(TAME_output)
            LT_results[order] = {
                "contraction runtime":{},
                "full runtime":{},
            }
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profile) in enumerate(TAME_output):

                #  aggregate the runtimes
                full_rt = reduce(elemwise_list_sum,[
                        profile["Matching Timings"],
                        profile["TAME_timings"],
                        profile["Scoring Timings"],
                        ])

                for (i,rt) in enumerate(full_rt):
                    params = all_shifts[i]
                    if params in LT_results[order]["full runtime"]:
                        LT_results[order]["full runtime"][params].append(rt)
                    else:
                        LT_results[order]["full runtime"][params] = [rt]

                for (i,rt) in enumerate(profile["TAME_timings"]):
                    params = all_shifts[i]
                    if params in LT_results[order]["contraction runtime"]:
                        LT_results[order]["contraction runtime"][params].append(rt)
                    else:
                        LT_results[order]["contraction runtime"][params] = [rt]
        
            for key in ["contraction runtime","full runtime"]:
                for (param,rts) in LT_results[order][key].items():
                    LT_results[order][key][param] = np.median(np.array(rts))


    #return LR_results, T_results

    #
    #  Create Plots 
    #
    if global_ax is None:
        fig = plt.figure(figsize=(5.5,3))
        global_ax = plt.gca()
        global_ax.set_axis_off()


    parity = 1  #used for checkerboard effect in plots.             
    

    linestyles= {
        'α:0.5_β:0.0':"dotted",
        'α:0.5_β:1.0':"solid",
        'α:1.0_β:0.0':(0,(3,1,1,1)),
        'α:1.0_β:1.0':(0,(5,1))
    }

    motif_label = 0
    rt_data = 1
    rank_data = 2
    A_motif_data = 3

    n = 4
    m = len(LT_results.keys())


    widths = [.5,3,2,1]
    col_width_ratios = [1]*m
    col_width_ratios.append(.8) 

                                # +1 for vioin legend
    gs = fig.add_gridspec(nrows=n,ncols=m + 1,hspace=0.1,wspace=0.1,
                          height_ratios=widths,width_ratios=col_width_ratios,
                          left=.18,right=.99,top=.925,bottom=.05)
                                        
    gs_ax = np.empty((n,m),object)
    iterate_tick_idx = 2


    # -- plot runtime data -- # 
    for idx,((LRT_order,LRT_data),(T_order,T_data),(LT_order,LT_data)) in enumerate(zip(LR_results.items(),T_results.items(),LT_results.items())):
    
        assert LRT_order == T_order
        assert LRT_order == LT_order

        order = LRT_order
        for row in range(n):

            ax = fig.add_subplot(gs[row,idx])
            gs_ax[row,idx] = ax


            if row == motif_label:
                ax.annotate(f"{order}",xy=(.5, .5), xycoords='axes fraction', c="k",size=10,ha="center",va="center")

            elif row == rt_data: 
                for (exp_data,c) in zip([LRT_data,T_data],[LRT_color,T_color]):
                    for (param,runtime) in exp_data["contraction runtime"].items():
                        if param == 'α:0.5_β:0.0' or param == 'α:1.0_β:1.0':
                            continue #ignore partial shifts
                        gs_ax[row,idx].plot(runtime,c=c,linestyle=linestyles[param])
    
                iterations = len(runtime)
                for (param,runtime) in LT_data["contraction runtime"].items():
                    if param == 'α:0.5_β:0.0' or param == 'α:1.0_β:1.0':
                            continue
                    gs_ax[row,idx].axhline(runtime/iterations,c=LT_color,linestyle=linestyles[param])
                

            elif row == rank_data:
                for (param,ranks) in LRT_data["ranks"].items():
                    if param == 'α:0.5_β:0.0' or param == 'α:1.0_β:1.0':
                        continue
                    if idx == iterate_tick_idx:
                        ax.set_zorder(3.5)
                    ax.plot(np.median(np.array(ranks),axis=0),c=[.1]*3,linestyle=linestyles[param])
                    #if T_order == 7:
                    #    for (motifs,rank) in sorted(zip(LRT_data["A motifs"],ranks),key=lambda x:x[0]):
                    #        print(f"motifs:{motifs}  ranks:{rank}")


            elif row == A_motif_data:
                make_violin_plot(ax,LRT_data["A motifs"],precision=0,c="k")


    # -- plot motif data -- # 

    #
    #  Adjust Axes
    #
    gs_ax[motif_label,0].set_ylabel(" Clique Size",rotation=0,labelpad=32.5,ha="center",va="center")
    
    subylabel_xpos = -.7 
    gs_ax[rt_data,0].set_ylabel("Contraction\nRuntime (s)",rotation=0,labelpad=32.5,ha="center",va="center")
    gs_ax[rt_data,0].annotate("25 unique\ntrials",xy=(subylabel_xpos, .15), xycoords='axes fraction',ha="center",fontsize=7,style='italic')
    
    for (idx,ax) in enumerate(gs_ax[rt_data,:]):

        ax.set_ylim(5e-5,5e3)
        ax.set_yscale("log")
        
        ax.yaxis.set_ticks_position('right')
        ax.set_yticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3])

        if idx == m-1:
            ax.set_yticklabels([r"$10^{-4}$",r"$10^{-3}$",None,r"$10^{-1}$",None,r"$10^1$",None,r"$10^3$"])
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0,6,14])
        ax.grid(True)
        ax.set_xlim(-1,15)

    bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.01)

    gs_ax[rt_data,3].annotate("LR-TAME",c=LRT_color,xy=(.225, .375), xycoords='axes fraction').set_bbox(bbox)
    gs_ax[rt_data,3].annotate("TAME",c=T_color,xy=(.2, .85), xycoords='axes fraction').set_bbox(bbox)
    gs_ax[rt_data,3].annotate(r"$\Lambda$"+"-TAME",c=LT_color,xy=(.225, .14), xycoords='axes fraction').set_bbox(bbox)

    gs_ax[rank_data,0].set_ylabel("Iterate\nRank",rotation=0,labelpad=32.5,ha="center",va="center")
    gs_ax[rank_data,0].annotate("max rank=100",xy=(subylabel_xpos, .15), xycoords='axes fraction',ha="center",fontsize=7,style='italic')
    
    for (idx,ax) in enumerate(gs_ax[rank_data,:].reshape(-1)):
        ax.set_ylim(-1,32.5)
        ax.yaxis.set_ticks_position('right')
        ax.set_yticks([0,1,5,10,15,20,25,30])
        if idx == m-1:
            ax.set_yticklabels([None,None,5,None,15,None,25,None])
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0,6,14])
        ax.grid(True)
        ax.set_xlim(-1,15)

    # -- add in shift annotations -- #

    gs_ax[rank_data,1].annotate('α=.5 β=1\n(both shifts)',xy=(.375, .65), xycoords='axes fraction',fontsize=6,ha="left",zorder=5)
    
    x_loc = .18
    gs_ax[rank_data,1].annotate('β=0 (no shifts)',xy=(x_loc, .1),xycoords='axes fraction',fontsize=6, ha="left",va="bottom")
    gs_ax[rank_data,1].annotate('α=1',xy=(x_loc - .01, .225), xycoords='axes fraction',fontsize=6,ha="left")
 



 
    legend_ax = fig.add_subplot(gs[A_motif_data,-1])
    make_violin_plot_legend(legend_ax)

    gs_ax[A_motif_data,0].set_ylabel("A motifs",rotation=0,labelpad=32.5,ha="center",va="center")
    gs_ax[A_motif_data,0].annotate("samples="+r"$10^4$",xy=(subylabel_xpos, .01), xycoords='axes fraction',ha="center",fontsize=7,style='italic')
    
    for ax in chain(gs_ax[[A_motif_data,motif_label],:].reshape(-1),[legend_ax]):
        ax.set_yticklabels([])

    for ax in chain(gs_ax.reshape(-1),[legend_ax]):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)
        ax.set_xticklabels([])

    #add back in right most column tick marks
    gs_ax[rt_data,m-1].tick_params(axis="y",direction="out",which='both', length=2)
    gs_ax[rank_data,m-1].tick_params(axis="y",direction="out",which='both', length=2)


    gs_ax[rank_data,iterate_tick_idx].xaxis.set_label_position("top")
    gs_ax[rank_data,iterate_tick_idx].xaxis.set_ticks_position('top')
    gs_ax[rank_data,iterate_tick_idx].tick_params(axis="x",direction="out", pad=-17.5,length=5)
    gs_ax[rank_data,iterate_tick_idx].set_xticklabels([1,5,15])
    bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=-.01)
    gs_ax[rank_data,iterate_tick_idx].annotate("iteration "+r"$(\ell)$",xy=(.5,1.25),ha="center",xycoords='axes fraction',fontsize=10)#.set_bbox(bbox)


    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def TAME_vs_LRTAME_clique_scaling_summarized(save_path=None):
    """This version stacks the results of LR-TAME and TAME on top of one another."""

    plt.rc('text.latex', preamble=r'\usepackage{/Users/charlie/Documents/Code/TKPExperimentPlots/latex/dgleich-math}\usepackage{amsmath}')

    #
    #  subplot routines 
    #
    extremal_tick_ypos = .2
        # subroutine globals
    def underline_text(ax,text,c,linestyle):
        tb = text.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        ax.annotate('', xy=(tb.xmin,tb.y0), xytext=(tb.xmax,tb.y0),             xycoords="figure fraction",arrowprops=dict(arrowstyle="-", color=c,linestyle=linestyle,linewidth=1.5,alpha=.8))

    def mark_as_algorithm(ax,text,c,linestyle,algorithm="LRTAME"):
        
        tb = text.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        
        # calculate asymmetry of x and y axes:
        x0, y0 = fig.transFigure.transform((0, 0)) # lower left in pixels
        x1, y1 = fig.transFigure.transform((1, 1)) # upper right in pixes
        dx = x1 - x0
        dy = y1 - y0
        maxd = max(dx, dy)

  

        if algorithm == "LRTAME":
                radius=.02
                height = radius * maxd / dy
                width = radius * maxd / dx

                p=ax.add_patch(patches.Ellipse((tb.xmin-.015,tb.y0+(5/8)*tb.height),width, height,color=LRT_color,transform=fig.transFigure,clip_on=False))
                
        elif algorithm == "TAME":
            side_length=.015
            height = side_length * maxd / dy
            width = side_length * maxd / dx

            p=ax.add_patch(patches.Rectangle((tb.xmax+.01,tb.y0+.5*(tb.height - side_length)),
                                        width, height,color=T_color,
                                        transform=fig.transFigure,clip_on=False))
        else:
            raise ValueError(f"algorithm must be either 'TAME' or 'LRTAME', got {algorithm}.\n")
    


    def make_violin_plot(ax,data,precision=2,c=None,v_alpha=.3,format="default",xlim=None,xscale="linear",column_type=None):

        bbox = dict(boxstyle="round", ec="w", fc="w", alpha=.5,pad=.01)
        if xscale=="linear":
            v = ax.violinplot(data,[.5], points=100, showmeans=False,widths=.15,
                        showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        elif xscale=="log":
            v = ax.violinplot(np.log10(data), points=100, showmeans=False,widths=.15,showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        else:
            raise ValueError(f"supports xscale values: 'linear' & 'log'; Got {xscale}.")


    

        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]

        newMedianLines = [[(x0,y1),(x0,y1+.7)]]
        v["cmedians"].set_segments(newMedianLines)

        # -- write data values as text
        
        if column_type is None:

            if format == "default":
                ax.annotate(f"{np.median(data):.{precision}f}",xy=(.475,.4),xycoords="axes fraction",ha="center",fontsize=10)#.set_bbox(bbox)
                ax.annotate(f"{np.min(data):.{precision}f}",xy=(.025,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
                ax.annotate(f"{np.max(data):.{precision}f}",xy=(.975,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
            elif format == "scientific":
                ax.annotate(f"{np.median(data):.{precision}e}",xy=(.5,.4),xycoords="axes fraction",ha="center",fontsize=10)
                ax.annotate(f"{np.min(data):.{precision}e}",xy=(.025,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
                ax.annotate(f"{np.max(data):.{precision}e}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
            else:
                print(f"expecting format to be either 'default' or 'scientific', got:{format}")
        elif column_type == "merged_axis":
            pass
        else:
            raise ValueError("column_type expecting 'merged_axis' or None, but got {column_type}\n")

        if c is not None:
            v["cmedians"].set_color(c)
            v["cmedians"].set_alpha(.75)
            for b in v['bodies']:
                # set colors 
                b.set_facecolor(c)
                b.set_edgecolor("None")            
                b.set_alpha(v_alpha)
                #b.set_color(c)

                #  -- only plot the top half of violin plot --  #
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)
                
                #  -- clip the top of the med-line to the top of the violin plot --  #
                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    #new_max_y += .04
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])
 
    def make_violin_plot_merged_axis(ax,data1,data2,c1,c2,marker1,marker2,format=None,**kwargs):

  
        make_violin_plot(ax,data1,**dict(kwargs,c=c1,column_type="merged_axis"))
        make_violin_plot(ax,data2,**dict(kwargs,format=format,c=c2,column_type="merged_axis"))


        # add markers to the median lines
        marker_size = 12.5
        marker_y_loc = ax.get_ylim()[-1]
        ax.scatter(np.log10(np.median(data1)),marker_y_loc,marker=T_marker,color=c1,s=marker_size)
        ax.scatter(np.log10(np.median(data2)),marker_y_loc,marker=LRT_marker,color=c2,s=marker_size)
        ax.set_ylim(0.9175, 1.1)
        #
        #ax.scatter(np.median(data1),.65,marker=marker1,s=5)
        bbox = dict(boxstyle="round", ec="w", fc="w", alpha=.5,pad=.01)
        min1 = np.min(data1)   
        min2 = np.min(data2)
        if min1 < min2:
            #text = f"{min1:.{kwargs['precision']}f}"
            #underlined_annotation(fig,ax,(.075,extremal_tick_ypos),text,linestyle=LRT_linestyle,ha="left",fontsize=8,alpha=.8)

            text = ax.annotate(f"{min1:.{kwargs['precision']}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8) 
            #mark_as_algorithm(ax,text,T_color,T_linestyle,algorithm="TAME")
            #underline_text(ax,text,T_color,T_linestyle) 
            """
            tb = text.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            ax.annotate('', xy=(tb.xmin,tb.y0), xytext=(tb.xmax,tb.y0),             xycoords="figure fraction",arrowprops=dict(arrowstyle="-", color='k',linestyle=T_linestyle))
            """
        else:
            text = ax.annotate(f"{min2:.{kwargs['precision']}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)  
            #mark_as_algorithm(ax,text,LRT_color,LRT_linestyle,algorithm="LRTAME")
            #underline_text(ax,text,LRT_color,LRT_linestyle)


        #minimum_val = min([np.min(data1),np.min(data2)])
        maximum_val = min([np.max(data1),np.max(data2)])
        max1 = np.max(data1)   
        max2 = np.max(data2)
        if max1 > max2:
            text = f"{maximum_val:.{kwargs['precision']}f}"
            text = ax.annotate(f"{max1:.{kwargs['precision']}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)  
            #mark_as_algorithm(ax,text,T_color,T_linestyle,algorithm="TAME")
            #underline_text(ax,text,T_color,T_linestyle)
            #underlined_annotation(fig,ax,(.925,extremal_tick_ypos),text,linestyle=LRT_linestyle,ha="right",fontsize=8,alpha=.8)
        else:
            text = ax.annotate(f"{maximum_val:.{kwargs['precision']}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
            #mark_as_algorithm(ax,text,LRT_color,LRT_linestyle,algorithm="LRTAME")
            #underline_text(ax,text,LRT_color,LRT_linestyle)

        ax.annotate(f"{np.median(data1):.{kwargs['precision']}f}",xy=(.7,.5),xycoords="axes fraction",ha="center",fontsize=10)#.set_bbox(bbox)
        ax.annotate(f"{np.median(data2):.{kwargs['precision']}f}",xy=(.3,.5),xycoords="axes fraction",ha="center",fontsize=10)#.set_bbox(bbox)



        #for x in sorted(dir(text)):
        #    print(x)
        """
        if format is None:        
            ax.annotate(f"{np.median(data1):.{kwargs[:precision]}f}",xy=(.5,.2),xycoords="axes fraction",ha="center",fontsize=10)
            ax.annotate(f"{np.min(data1):.{precision}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
            ax.annotate(f"{np.max(data1):.{precision}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
        elif format == "scientific":
            ax.annotate(f"{np.median(data):.{precision}e}",xy=(.5,.2),xycoords="axes fraction",ha="center",fontsize=10)
            ax.annotate(f"{np.min(data):.{precision}e}",xy=(.025,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
            ax.annotate(f"{np.max(data):.{precision}e}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
        else:
            print(f"expecting format to be 'scientific' or None, got:{format}")
        """

    def make_violin_plot_legend(ax,c="k"):
        
        np.random.seed(12)#Ok looking:12
        v = ax.violinplot([np.random.normal() for i in range(50)], points=100, positions=[.6], showmeans=False, 
                        showextrema=False, showmedians=True,widths=.6,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #ax.axes.set_axis_off()
        ax.set_ylim(.5,1.0)
        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        #newMedianLines = [[(x0,y0-.125),(x0,y0 + (y1-y0)/2 -.1)]]#,[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
        newMedianLines = [[(x0,y1 +.05),(x0,1.5)]]
        v["cmedians"].set_segments(newMedianLines)

        bbox = dict(boxstyle="round", ec="w", fc="w", alpha=.3,pad=.01)
        ax.annotate("median",xy=(.5,.375),xycoords="axes fraction",ha="center",va="center",fontsize=10)#.set_bbox(bbox)
        ax.annotate(f"min",xy=(.025,-.125),xycoords="axes fraction",ha="left",fontsize=9,alpha=.8)
        ax.annotate(f"max",xy=(.975,-.125),xycoords="axes fraction",ha="right",fontsize=9,alpha=.8)

        if c is not None:
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor("None")
                b.set_alpha(.3)
                
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)

                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    new_max_y += .02
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])

    def make_merged_violin_plot_legend(ax):
        
        np.random.seed(12)#Ok looking:12
        data1 = [np.random.normal(-.25,.25) for i in range(50)]
        v1= ax.violinplot(data1, points=100, positions=[.6], showmeans=False, showextrema=False, showmedians=True,widths=.6,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        data2 = [np.random.normal(.5,.25) for i in range(50)]
        v2 = ax.violinplot(data2, points=100, positions=[.6], 
                          showmeans=False, showextrema=False, showmedians=True,widths=.6,vert=False)
        ax.set_ylim(.5,1.0)

        marker_size = 12.5
        marker_y_loc = ax.get_ylim()[-1] - .05
        ax.scatter(-0.282,marker_y_loc,marker=LRT_marker,color=LRT_color,s=marker_size)
        ax.scatter(np.log10(np.median(data2))+.795,marker_y_loc,marker=T_marker,color=T_color,s=marker_size)

        bbox = dict(boxstyle="sawtooth", ec="w", fc="w", alpha=.3,pad=-.15)


        for (c,v) in [(LRT_color,v1),(T_color,v2)]:
            #  --  update median lines to have a gap  --  #
            ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
            #newMedianLines = [[(x0,y0-.125),(x0,y0 + (y1-y0)/2 -.1)]]#,[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
            newMedianLines = [[(x0,y1 +.05),(x0,1.5)]]
            v["cmedians"].set_segments(newMedianLines)

            med_label1 = ax.annotate("LR-TAME med.",xy=(.275,.35),xycoords="axes fraction",ha="center",va="center",fontsize=9)#.set_bbox(bbox)
            #ax.annotate("median",xy=(.25,.35),xycoords="axes fraction",ha="center",va="top",fontsize=9)#.set_bbox(bbox)
            #mark_as_algorithm(ax,med_label1,LRT_color,LRT_linestyle,algorithm="LRTAME")
            med_label2 = ax.annotate("TAME med.",xy=(.725,.35),xycoords="axes fraction",ha="center",va="center",fontsize=9)#.set_bbox(bbox)
            #ax.annotate("median",xy=(.7,.35),xycoords="axes fraction",ha="center",va="top",fontsize=9)#.set_bbox(bbox)
            #mark_as_algorithm(ax,med_label2,T_color,T_linestyle,algorithm="TAME")
            min_label = ax.annotate(f"min",xy=(.075,-.125),xycoords="axes fraction",ha="left",fontsize=9,alpha=.8)
            #mark_as_algorithm(ax,min_label,LRT_color,LRT_linestyle,algorithm="LRTAME")
            
            max_label = ax.annotate(f"max",xy=(.925,-.125),xycoords="axes fraction",ha="right",fontsize=9,alpha=.8)
            #mark_as_algorithm(ax,max_label,T_color,T_linestyle,algorithm="TAME")
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor("None")
                b.set_alpha(.3)
                #b.set_color(c)
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)

                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    new_max_y += .02
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])



    

    #
    #  Parse Data
    #

    results_path = TAME_RESULTS + "RG_DupNoise/"

    elemwise_list_sum =lambda l1,l2: [a + b for (a,b) in zip(l1,l2)]
    filename = "LowRankTAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7,8]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"

    with open(results_path + filename,"r") as file:
        data = json.load(file)
        LR_results = {}
        debug_results = {}
        LR_seeds = {}

        for (order, LRTAME_output) in data:
            trials = len(LRTAME_output)
            LR_results[order] = {
                "full runtime":{},
                "contraction runtime":{},
                "A motifs":[],
                "ranks":{},
            }
            debug_results[order] = {
                "runtimes":[],
                "ranks":[]
            }
            LR_seeds[order] = []
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profiling) in enumerate(LRTAME_output):
                LR_results[order]["A motifs"].append(A_motifs)
                LR_seeds[order].append(seed)
                for (params,profile) in profiling:
                    contract_rt = reduce(elemwise_list_sum,[
                        profile["low_rank_factoring_timings"],
                        profile["contraction_timings"],
                        ])
                    if params in LR_results[order]["contraction runtime"]:
                        LR_results[order]["contraction runtime"][params].append(np.max(contract_rt))
                    else:
                        LR_results[order]["contraction runtime"][params] = [np.max(contract_rt)]


                    if params == 'α:0.5_β:1.0':
                        debug_results[order]["ranks"].append(profile["ranks"])
                        debug_results[order]["runtimes"].append(contract_rt)

                    ranks = profile["ranks"]
                    if len(ranks) < 15:
                        # if algorithm terminated from tol bounds, extend last rank to fill the rest
                        ranks.extend([ranks[-1]]*(15-len(ranks)))

                    if params in LR_results[order]["ranks"]:
                        LR_results[order]["ranks"][params].append(profile["ranks"])
                    else:
                        LR_results[order]["ranks"][params] = [profile["ranks"]]

    #return debug_results

    file = "TAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"
    with open(results_path + file,"r") as file:
        data = json.load(file)

        T_results = {}


        for (order, TAME_output) in data:
            trials = len(TAME_output)
            T_results[order] = {
                "contraction runtime":{},
                "full runtime":{},
            }
 
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profiling) in enumerate(TAME_output):
                for (params,profile) in profiling:
               
                    if params in T_results[order]["contraction runtime"]:
                        T_results[order]["contraction runtime"][params].append(np.max(profile["contraction_timings"]))
                    else:
                        T_results[order]["contraction runtime"][params] = [np.max(profile["contraction_timings"])]


    file = "LambdaTAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"
    with open(results_path + file,"r") as file:
        data = json.load(file)
        LT_results = {}

        all_shifts = [
            'α:0.5_β:0.0',
            'α:0.5_β:1.0',
            'α:1.0_β:0.0',
            'α:1.0_β:1.0',
        ]
        for (order, TAME_output) in data:
            trials = len(TAME_output)
            LT_results[order] = {
                "contraction runtime":{},
                "full runtime":{},
            }
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profile) in enumerate(TAME_output):

                #  aggregate the runtimes
                full_rt = reduce(elemwise_list_sum,[
                        profile["Matching Timings"],
                        profile["TAME_timings"],
                        profile["Scoring Timings"],
                        ])

                for (i,rt) in enumerate(full_rt):
                    params = all_shifts[i]
                    if params in LT_results[order]["full runtime"]:
                        LT_results[order]["full runtime"][params].append(rt)
                    else:
                        LT_results[order]["full runtime"][params] = [rt]

                for (i,rt) in enumerate(profile["TAME_timings"]):
                    params = all_shifts[i]
                    if params in LT_results[order]["contraction runtime"]:
                        LT_results[order]["contraction runtime"][params].append(rt)
                    else:
                        LT_results[order]["contraction runtime"][params] = [rt]

    #
    #  Create Plots 
    #
    fig = plt.figure(figsize=(5,3))
    global_ax = plt.gca()
    global_ax.set_axis_off()


    parity = 1  #used for checkerboard effect in plots.             
    

    linestyles= {
        'α:0.5_β:0.0':"dotted",
        'α:0.5_β:1.0':"solid",
        'α:1.0_β:0.0':(0,(3,1,1,1)),
        'α:1.0_β:1.0':(0,(5,1))
    }

    # column assignment

    motif_label = 0
    A_motif_data = 1
    rank_data = 2
    TAME_rt = 3
    LRTAME_rt = 3
    LTAME_rt = 5

    #rt_data = 1

    main_gs = fig.add_gridspec(2, 1,hspace=0.0,wspace=0.0, height_ratios = [1,.15],
                                left=0.05,right=.975,top=.85,bottom=0.025)

    n = len(T_results.keys())
    m = 4 # - 1 for no LT 
          # - 1 for merging TAME w/ LRT
    
    heights = [.1,.3,.3,.7]
    row_height_ratios = [1]*n 
    #col_width_ratios.append(.8) 


    
    gs = main_gs[0].subgridspec(nrows=n,ncols=m,hspace=0.0,wspace=0.1, 
                          width_ratios=heights,height_ratios=row_height_ratios)
    legend_gs = main_gs[1].subgridspec(nrows=1,ncols=20)
                       
    gs_ax = np.empty((n,m),object)

    iterate_tick_idx = 2


    # -- plot runtime data -- # 
    LambdaTAME_runtime_data = []

    motif_label_xloc = .4
    for idx,((LRT_order,LRT_data),(T_order,T_data),(LT_order,LT_data)) in enumerate(zip(LR_results.items(),T_results.items(),LT_results.items())):
        assert LRT_order == T_order
        assert LRT_order == LT_order

        LambdaTAME_runtime_data.append((T_order,np.max(LT_data["contraction runtime"]['α:0.5_β:1.0'])))

        order = LRT_order
        for col in range(m):

            if idx == 0:
                ax = fig.add_subplot(gs[idx,col])
            else:
                ax = fig.add_subplot(gs[idx,col],sharex=gs_ax[0,col])
            
            gs_ax[idx,col] = ax


            if col == motif_label:
                ax.annotate(f"{order}",xy=(motif_label_xloc, .5), xycoords='axes fraction', c="k",size=10,ha="center",va="center")#,weight="bold")
            elif col == TAME_rt:
                #ax.set_xscale("log")
                make_violin_plot_merged_axis(ax,T_data["contraction runtime"]['α:0.5_β:1.0'],
                                                LRT_data["contraction runtime"]['α:0.5_β:1.0'],
                                                T_color,LRT_color,T_marker,LRT_marker, precision=2,v_alpha=.3,xscale="log")


            elif col == LRTAME_rt:
                #ax.set_xscale("log")
                pass
              
            elif col == LTAME_rt:
                make_violin_plot(ax,LT_data["contraction runtime"]['α:0.5_β:1.0'],
                                 precision=1,c=LT_color,v_alpha=.3,format = "scientific",xscale="log")


            elif col == A_motif_data:
                make_violin_plot(ax,LRT_data["A motifs"],precision=0,c="k",v_alpha=.3)

            elif col == rank_data:
                make_violin_plot(ax,np.array(LRT_data["ranks"]['α:0.5_β:1.0']).max(axis=1),precision=0,c="k",v_alpha=.3)
                #ax.set_xlim(13,61)


    print(f"(order,LambdTAME maximum contraction runtimes):\n{LambdaTAME_runtime_data}")
        

    vp_legend_ax = fig.add_subplot(legend_gs[3:8])
    make_violin_plot_legend(vp_legend_ax)

    merged_vp_legend_ax = fig.add_subplot(legend_gs[11:20])
    make_merged_violin_plot_legend(merged_vp_legend_ax)

    #marker_legend_ax = fig.add_subplot(legend_gs[6:10])
    label_font = 11
    #LRT_label = marker_legend_ax.annotate("LR-TAME",rotation=0,ha="right",va="top",xy=(.85, 1.0), xycoords='axes fraction',c=LRT_color,fontsize=label_font)
    #mark_as_algorithm(marker_legend_ax,LRT_label,LRT_color,LRT_linestyle,"LRTAME")

    #TAME_label = marker_legend_ax.annotate("TAME",rotation=0,ha="right",va="bottom",xy=(.85, 0.0), xycoords='axes fraction',c=T_color,fontsize=label_font)
    #mark_as_algorithm(marker_legend_ax ,TAME_label,T_color,T_linestyle,"TAME")


    #make_violin_plot_legend(merged_vp_legend_ax)
    # -- create legends -- # 



    #
    #  Adjust Axes
    #
    
    # -- Set the column titles -- #
    title_ypos = 1.1 
    annotation_ypos = .6
    gs_ax[0,motif_label].annotate("Clique\nSize",ha="center",va="bottom",xy=(motif_label_xloc, title_ypos), xycoords='axes fraction')
    
    
 
    #bbox = dict(boxstyle="round", ec=checkboard_color, fc=checkboard_color, alpha=1.0,pad=-.1)
    
    #gs_ax[0,LRTAME_rt].set_ylabel("Max Iterate\nTTV Time (s)",rotation=0,labelpad=30,ha="center",va="center")#,c=LT_color,xy=(.3, .125), xycoords='axes fraction').set_bbox(bbox)
    gs_ax[0,LRTAME_rt].annotate("Longest\nContraction (s)",rotation=0,ha="center",va="bottom",xy=(.5, title_ypos), xycoords='axes fraction')#,c=LT_color,xy=(.3, .125), xycoords='axes fraction').set_bbox(bbox)
   

    gs_ax[0,rank_data].annotate("Max (LR-)TAME\nIterate Rank",ha="center",va="bottom",xy=(.5, title_ypos), xycoords='axes fraction')

    gs_ax[0,A_motif_data].annotate("A motifs",xy=(.5, title_ypos), xycoords='axes fraction',ha="center",va="bottom")

    #y_pos = -1.25
    #gs_ax[-1,-1].annotate("LRT - LowRankTAME",xy=(.625, y_pos), xycoords='axes fraction',ha="right",va="center",fontsize=11,c=LRT_color)
    #gs_ax[-1,-1].annotate("T - TAME",xy=(1.0, y_pos), xycoords='axes fraction',ha="right",va="center",fontsize=11,c=T_color)#
    # -- make a label for shared x-axis
    super_title_ypos = .875


    additional_ax = [
        global_ax,vp_legend_ax,merged_vp_legend_ax,#marker_legend_ax #legend_ax # annotation_ax,
    ]

    for ax in chain(gs_ax.reshape(-1),additional_ax):
        ax.set_yticklabels([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)
        ax.set_xticklabels([])

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



#
#  LVGNA Alignments
#

def LVGNA_end_to_end_relative_to_TAME_table_with_microplots(save_path=None):

    fig = plt.figure(figsize=(5,4.2))
    

    #
    #    Subroutines
    #

    def make_microplot_legend(ax):
        ax.grid(True)
        ax.set_ylim(-.1,1.1)
        ax.set_xlim(-.1,1.1)
        ax.set_xticks(np.linspace(0,1,5))
        ax.set_yticks(np.linspace(0,1,5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel(r"$|T_A||T_B|$",labelpad=-2)
        ax.set_ylabel(r"Data",labelpad=-2)

        bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1,pad=0.0)
        ax.annotate("log",xy=(.4,.1),xycoords="axes fraction",va="center",fontsize=6).set_bbox(bbox)
        ax.annotate("log",xy=(.1,.4),xycoords="axes fraction",ha="center",rotation=90,fontsize=6).set_bbox(bbox)

    def make_violin_plot_legend(ax,c="k"):
        
        np.random.seed(12)#Ok looking:12
        v = ax.violinplot([np.random.normal() for i in range(50)], points=100, positions=[.6], showmeans=False, 
                        showextrema=False, showmedians=True,widths=.6,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #ax.axes.set_axis_off()
        ax.set_ylim(.5,1.0)
        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        #newMedianLines = [[(x0,y0-.125),(x0,y0 + (y1-y0)/2 -.1)]]#,[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
        newMedianLines = [[(x0,y1 +.05),(x0,1.5)]]
        v["cmedians"].set_segments(newMedianLines)

        ax.annotate("median",xy=(.5,.35),xycoords="axes fraction",ha="center",va="center",fontsize=10)
        ax.annotate(f"min",xy=(.025,0),xycoords="axes fraction",ha="left",fontsize=9,alpha=.8)
        ax.annotate(f"max",xy=(.975,0),xycoords="axes fraction",ha="right",fontsize=9,alpha=.8)

        if c is not None:
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor("None")
                b.set_alpha(.3)
                #b.set_color(c)
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)

                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    new_max_y += .02
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])


    def underline_text(ax,renderer,text,c,linestyle):
        tb = text.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        ax.annotate('', xy=(tb.xmin,tb.y0), xytext=(tb.xmax,tb.y0),             xycoords="figure fraction",arrowprops=dict(arrowstyle="-", color=c,linestyle=linestyle,linewidth=1.5,alpha=.8))


    def make_violin_plot(ax,data,precision=2,c=None,v_alpha=.8,format="default",xlim=None,xscale="linear",column_type=None):

       
        #background_v = ax.violinplot(data, points=100, positions=[0.5], showmeans=False, 
        #                showextrema=False, showmedians=False,widths=.5,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #
        #positions=[0.5], ,widths=.5
        if xscale=="linear":
            v = ax.violinplot(data,[.5], points=100, showmeans=False,widths=.15,
                        showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        elif xscale=="log":
            v = ax.violinplot(np.log10(data), points=100, showmeans=False,widths=.15,showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        else:
            raise ValueError(f"supports xscale values: 'linear' & 'log'; Got {xscale}.")
        #ax.set_ylim(0.95, 1.3)
        #ax.set_xlim(np.min(data),np.max(data))



        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]

        newMedianLines = [[(x0,y1),(x0,y1+.7)]]
        v["cmedians"].set_segments(newMedianLines)

        # -- place extremal markers underneath
        """
        v['cbars'].set_segments([]) # turns off x-axis spine
        for segment in [v["cmaxes"],v["cmins"]]:
            ((x,y0),(_,y1)) = segment.get_segments()[0]
            segment.set_segments([[(x,0.45),(x,.525)]])
            segment.set_color(c)
        """

        # -- write data values as text
        bbox = dict(boxstyle="round", ec="w", fc="w", alpha=.5,pad=.05)
        extremal_tick_ypos = .25
        if column_type is None:

            if format == "default":
                ax.annotate(f"{np.median(data):.{precision}f}",xy=(.5,.4),xycoords="axes fraction",ha="center",fontsize=10)#.set_bbox(bbox)
                ax.annotate(f"{np.min(data):.{precision}f}",xy=(.025,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)#.set_bbox(bbox)
                ax.annotate(f"{np.max(data):.{precision}f}",xy=(.975,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)#.set_bbox(bbox)
            elif format == "scientific":
                ax.annotate(f"{np.median(data):.{precision}e}",xy=(.5,.4),xycoords="axes fraction",ha="center",fontsize=10)#.set_bbox(bbox)
                ax.annotate(f"{np.min(data):.{precision}e}",xy=(.025,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)#.set_bbox(bbox)
                ax.annotate(f"{np.max(data):.{precision}e}",xy=(.975,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)#.set_bbox(bbox)
            else:
                print(f"expecting format to be either 'default' or 'scientific', got:{format}")
        elif column_type == "merged_axis":
            pass
        else:
            raise ValueError("column_type expecting 'merged_axis' or None, but got {column_type}\n")

        if c is not None:
            v["cmedians"].set_color(c)
            v["cmedians"].set_alpha(.75)
            for b in v['bodies']:
                # set colors 
                b.set_facecolor(c)
                b.set_edgecolor("None")            
                b.set_alpha(v_alpha)
                #b.set_color(c)

                #  -- only plot the top half of violin plot --  #
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)
                
                #  -- clip the top of the med-line to the top of the violin plot --  #
                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    #new_max_y += .04
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])
                #clip_to_top_of_violin(v["cmaxes"])
                #clip_to_top_of_violin(v["cmins"])
 
    #
    #    Load and Parse Pre + Post Processing information. 
    #
    

    TAME_tri_results,TAME_edge_match,\
    TAME_pp_edge_match,TAME_pp_tri_match,\
    TAME_runtimes, TAME_impTTV_runtimes, TAME_BM_runtimes,\
    TAME_pp_runtime, exp_idx = CT_LP.parse_results()
    full_TAME_runtimes = TAME_runtimes + TAME_pp_runtime

    graph_names = [" ".join(file.split(".smat")[0].split("_")) for (file,idx) in sorted(exp_idx.items(),key=lambda x:x[1])]
    gn_dict = {graph:i for (i,graph) in enumerate(graph_names)}

    LGRAAL_graphs, LGRAAL_tri_results, LGRAAL_runtimes, _,_, _,_ = get_results()
    LGRAAL_perm = [gn_dict[graph] for graph in LGRAAL_graphs]

    LGRAAL_tri_results = LGRAAL_tri_results[np.ix_(LGRAAL_perm, LGRAAL_perm)]
    LGRAAL_runtimes = LGRAAL_runtimes[np.ix_(LGRAAL_perm, LGRAAL_perm)]


    #data_path = TAME_RESULTS + "LVGNA_Experiments/klauPostProcessing/"
    data_path = TAME_RESULTS + "LVGNA_Alignments/"

    file = "LVGNA_pairwiseAlignment_LambdaTAME_MultiMotif_alphas:[1.0,0.5]_betas:[0.0,1.0,10.0,100.0]_order:3_samples:3000000_data.json"
    file = "LVGNA_pairwiseAlignment_LambdaTAME_alphas:[0.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_postProcessing:KlauAlgo_profile:true_tol:1e-6_results.json"
    with open(data_path + file,"r") as f:
        data = json.load(f)
        vertex_products = []
        edge_products = []
        motif_products = []
        LT_klau_tri_match_ratios = []
        LT_klau_edge_match_ratios = []
        LT_klau_runtimes = []
        #return data
        #for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LT_profile,LT_edges_matched,klau_edges_matched,klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity) in data:
        for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,LT_profile,LT_edges_matched,klau_edges_matched,klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity,f_status) in data:
            
            i = gn_dict[" ".join(file_i.split(".smat")[0].split("_"))]
            j = gn_dict[" ".join(file_j.split(".smat")[0].split("_"))]


            pre_processed_time = sum([sum(v) for (k,v) in  LT_profile.items() if "Timings" in k])
            LT_klau_runtimes.append(full_TAME_runtimes[i,j]/(pre_processed_time+klau_setup_rt+klau_rt))

            vertex_A = vertex_counts[" ".join(file_i.split(".smat")[0].split("_"))]
            vertex_B = vertex_counts[" ".join(file_j.split(".smat")[0].split("_"))]
            vertex_products.append(vertex_A*vertex_B)
 

            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            edge_products.append(edges_A*edges_B)

            #motif_products.append(A_Motifs[0]*B_Motifs[0])
            motif_products.append(A_Motifs*B_Motifs)
      
            LT_klau_edge_match_ratios.append((klau_edges_matched/min(edges_A,edges_B))/TAME_pp_edge_match[i,j])
            #LT_klau_tri_match_ratios.append((klau_tris_matched/min(A_Motifs[0],B_Motifs[0]))/TAME_pp_tri_match[i,j])
            LT_klau_tri_match_ratios.append((klau_tris_matched/min(A_Motifs,B_Motifs))/TAME_pp_tri_match[i,j])
    """
    #with open(data_path + "LVGNA_pairwiseAlignment_LambdaTAME_MultiMotif_alphas:[1.0,0.5]_betas:[0.0,1.0,10.0,100.0]_order:3_postProcessing:SuccessiveKlau_samples:3000000_data.json","r") as f:
    with open(data_path + "LVGNA_pairwiseAlignment_LambdaTAME_MultiMotif_alphas:[1.0,0.5]_betas:[0.0,1.0,10.0,100.0]_order:3_postProcessing:SuccessiveKlau-const-iter:5-maxIter:500_samples:10000000_data.json","r") as f:
        data = json.load(f)
        successive_klau_tri_match_ratios = []
        successive_klau_edge_match_ratios = []
        successive_klau_runtimes = []
     
        for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LT_profile,LT_edges_matched,sklau_edges_matched,sklau_tris_matched,_,successive_klau_profiling) in data:
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            
            successive_klau_edge_match_ratios.append(sklau_edges_matched/min(edges_A,edges_B))
            successive_klau_tri_match_ratios.append(sklau_tris_matched/min(A_Motifs[0],B_Motifs[0]))
            successive_klau_runtimes.append(sum(successive_klau_profiling["runtime"]))
    """
    file = "LVGNA_pairwiseAlignment_LambdaTAME_MultiMotif_alphas:[1.0,0.5]_betas:[0.0,1.0,10.0,100.0]_order:3_postProcessing:TabuSearch_samples:30000000_data.json"
    file = "LVGNA_pairwiseAlignment_LambdaTAME_alphas:[0.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_postProcessing:LocalSearch_profile:true_tol:1e-6_results.json"
    with open(data_path + file,"r") as f:
   
        data = json.load(f)
        LT_tabu_tri_match_ratios = []
        LT_tabu_edge_match_ratios = []
        LT_tabu_runtimes = []
        #return data
        #for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LT_profile,LT_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,tabu_full_rt) in data:
        for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,LT_profile,LT_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,tabu_full_rt) in data:
        
            i = gn_dict[" ".join(file_i.split(".smat")[0].split("_"))]
            j = gn_dict[" ".join(file_j.split(".smat")[0].split("_"))]

           
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            
            LT_tabu_edge_match_ratios.append((tabu_edges_matched/min(edges_A,edges_B))/TAME_pp_edge_match[i,j])

            #LT_tabu_tri_match_ratios.append((tabu_tris_matched/min(A_Motifs[0],B_Motifs[0]))/TAME_pp_tri_match[i,j])
            LT_tabu_tri_match_ratios.append((tabu_tris_matched/min(A_Motifs,B_Motifs))/TAME_pp_tri_match[i,j])
     
            pre_processed_time = sum([sum(v) for (k,v) in  LT_profile.items() if "Timings" in k])
            LT_tabu_runtimes.append(full_TAME_runtimes[i,j]/(pre_processed_time + tabu_full_rt))

    file = "LVGNA_pairwiseAlignment_LowRankTAME_alphas:[0.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_order:3_postProcessing:KlauAlgo_profile:true_samples:3000000_tol:1e-6_results.json"
    file = "LVGNA_pairwiseAlignment_LowRankTAME_alphas:[0.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_postProcessing:KlauAlgo_profile:true_tol:1e-6_results_colMajor.json"
    with open(data_path + file,"r") as f:
        data = json.load(f)
        #add in missing \beta = 1.0 case
  
        LRT_klau_tri_match_ratios = []
        LRT_klau_edge_match_ratios = []
        LRT_klau_runtimes = []

        #for (file_i,file_j,LRT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LRT_profile,LRT_edges_matched,klau_edges_matched,klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity,f_status) in data:
        for (file_i,file_j,LRT_matchedMotifs,A_Motifs,B_Motifs,LRT_profile,LRT_edges_matched,klau_edges_matched,klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity,f_status) in data:
            
            i = gn_dict[" ".join(file_i.split(".smat")[0].split("_"))]
            j = gn_dict[" ".join(file_j.split(".smat")[0].split("_"))]
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            
            LRT_klau_edge_match_ratios.append((klau_edges_matched/min(edges_A,edges_B))/TAME_pp_edge_match[i,j])

            LRT_klau_tri_match_ratios.append((klau_tris_matched/min(A_Motifs,B_Motifs))/TAME_pp_tri_match[i,j])


            pre_processed_time = 0.0
            for (param, profile) in LRT_profile:
                pre_processed_time += sum([sum(v) for (k,v) in profile.items() if "timings" in k])
            post_processed_time = klau_setup_rt + klau_rt
            LRT_klau_runtimes.append(full_TAME_runtimes[i,j]/(pre_processed_time + post_processed_time))

    
    file = "LVGNA_pairwiseAlignment_LowRankTAME_alphas:[1.0,0.5]_betas:[0.0,1.0,10.0,100.0]_iter:15_order:3_postProcessing:TabuSearch_profile:true_tol:1e-6_results.json"
    file = "LVGNA_pairwiseAlignment_LowRankTAME_alphas:[0.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_postProcessing:LocalSearch_profile:true_tol:1e-6_results_colMajor.json"
    
    with open(data_path +file ,"r") as f:
        data = json.load(f)

        LRT_tabu_tri_match_ratios = []
        LRT_tabu_edge_match_ratios = []
        LRT_tabu_runtimes = []

        #for (file_i,file_j,LRT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LRT_profile,LRT_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,tabu_full_rt)  in data:
        for (file_i,file_j,LRT_matchedMotifs,A_Motifs,B_Motifs,LRT_profile,LRT_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,tabu_full_rt)  in data:
            
            i = gn_dict[" ".join(file_i.split(".smat")[0].split("_"))]
            j = gn_dict[" ".join(file_j.split(".smat")[0].split("_"))]
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            
            #print(f"{LRT_matchedMotifs}/min({A_Motifs},{B_Motifs})")

            LRT_tabu_edge_match_ratios.append((tabu_edges_matched/min(edges_A,edges_B))/TAME_pp_edge_match[i,j])

            LRT_tabu_tri_match_ratios.append((tabu_tris_matched/min(A_Motifs,B_Motifs))/TAME_pp_tri_match[i,j])


            pre_processed_time = 0.0
            for (param, profile) in LRT_profile:
                pre_processed_time += sum([sum(v) for (k,v) in profile.items() if "timings" in k])
            LRT_tabu_runtimes.append(full_TAME_runtimes[i,j]/(pre_processed_time + tabu_full_rt))

    """
    file = "LVGNA_pairwiseAlignment_LowRankTAME_alphas:[0.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_postProcessing:LocalSearch_profile:true_tol:1e-6_results_colMajor.json"
    with open(data_path +file ,"r") as f:
        data = json.load(f)
        for (file_i,file_j,LRT_matchedMotifs,A_Motifs,B_Motifs,LRT_profile,LRT_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,tabu_full_rt)  in data:
            #print(f"{LRT_matchedMotifs}/min({A_Motifs},{B_Motifs})")
            #print(LRT_profile[0]["ranks"])
    """

    with open(data_path + "LVGNA_pairwiseAlignment_LowRankTAME_lrm_alphas:[0.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_order:3_postProcessing:KlauAlgo_profile:true_samples:3000000_tol:1e-6_results.json",'r') as f:
        data = json.load(f)
  
        LRT_lrm_klau_tri_match_ratios = []
        LRT_lrm_klau_edge_match_ratios = []
        LRT_lrm_klau_runtimes = []

        for (file_i,file_j,LRT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LRT_profile,LRT_edges_matched,klau_edges_matched,klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity,f_status) in data:
            
            i = gn_dict[" ".join(file_i.split(".smat")[0].split("_"))]
            j = gn_dict[" ".join(file_j.split(".smat")[0].split("_"))]
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            
            LRT_lrm_klau_edge_match_ratios.append((klau_edges_matched/min(edges_A,edges_B))/TAME_pp_edge_match[i,j])

            LRT_lrm_klau_tri_match_ratios.append((klau_tris_matched/min(A_Motifs,B_Motifs))/TAME_pp_tri_match[i,j])


            pre_processed_time = 0.0
            for (param, profile) in LRT_profile:
                pre_processed_time += sum([sum(v) for (k,v) in profile.items() if "timings" in k])
            post_processed_time = klau_setup_rt + klau_rt
            LRT_lrm_klau_runtimes.append(full_TAME_runtimes[i,j]/(pre_processed_time + post_processed_time))


    with open(data_path + "LVGNA_pairwiseAlignment_LowRankTAME_lrm_alphas:[0.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_order:3_postProcessing:TabuSearch_profile:true_samples:3000000_tol:1e-6_results.json",'r') as f:
        data = json.load(f)
        #add in missing \beta = 1.0 case
  
        LRT_lrm_tabu_tri_match_ratios = []
        LRT_lrm_tabu_edge_match_ratios = []
        LRT_lrm_tabu_runtimes = []

        for (file_i,file_j,LRT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LRT_profile,LRT_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,tabu_full_rt)  in data:
            
            i = gn_dict[" ".join(file_i.split(".smat")[0].split("_"))]
            j = gn_dict[" ".join(file_j.split(".smat")[0].split("_"))]
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            
            LRT_lrm_tabu_edge_match_ratios.append((tabu_edges_matched/min(edges_A,edges_B))/TAME_pp_edge_match[i,j])
            LRT_lrm_tabu_tri_match_ratios.append((tabu_tris_matched/min(A_Motifs,B_Motifs))/TAME_pp_tri_match[i,j])

            pre_processed_time = 0.0
            for (param, profile) in LRT_profile:
                pre_processed_time += sum([sum(v) for (k,v) in profile.items() if "timings" in k])
            LRT_lrm_tabu_runtimes.append(full_TAME_runtimes[i,j]/(pre_processed_time + tabu_full_rt))


    with open(data_path + "LVGNA_pairwiseAlignment_LowRankEigenAlign_postProcessing:TabuSearch_profile:true_results.json") as f:
        data = json.load(f)
        #for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,LREA_profile,LREA_edges_matched,LREA_klau_edges_matched,LREA_klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity,f_status) in data:
        LREA_tabu_tri_match_ratios = []
        LREA_tabu_edge_match_ratios = []
        LREA_tabu_runtimes = []
        for (file_i,file_j,LREA_matchedMotifs,A_Motifs,B_Motifs,LREA_runtime,LREA_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,tabu_full_rt) in data:
                
                i = gn_dict[" ".join(file_i.split(".smat")[0].split("_"))]
                j = gn_dict[" ".join(file_j.split(".smat")[0].split("_"))]

                edges_A = edge_counts[file_i.split(".smat")[0]]
                edges_B = edge_counts[file_j.split(".smat")[0]]
                
                LREA_tabu_edge_match_ratios.append((tabu_edges_matched/min(edges_A,edges_B))/TAME_pp_edge_match[i,j])
      

                LREA_tabu_tri_match_ratios.append((tabu_tris_matched/min(A_Motifs,B_Motifs))/TAME_pp_tri_match[i,j])


                LREA_tabu_runtimes.append(full_TAME_runtimes[i,j]/(LREA_runtime+tabu_full_rt))


    with open(data_path + "LVGNA_pairwiseAlignment_LowRankEigenAlign_postProcessing:KlauAlgo_profile:true_results.json") as f:
        data = json.load(f)
        LREA_klau_tri_match_ratios = []
        LREA_klau_edge_match_ratios = []
        LREA_klau_runtimes = []
        for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,LREA_runtime,LREA_edges_matched,LREA_klau_edges_matched,LREA_klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity,f_status) in data:
            i = gn_dict[" ".join(file_i.split(".smat")[0].split("_"))]
            j = gn_dict[" ".join(file_j.split(".smat")[0].split("_"))]
            
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            

            LREA_klau_edge_match_ratios.append((LREA_klau_edges_matched/min(edges_A,edges_B))/TAME_pp_edge_match[i,j])
            LREA_klau_tri_match_ratios.append((LREA_klau_tris_matched/min(A_Motifs,B_Motifs))/TAME_pp_tri_match[i,j])
            LREA_klau_runtimes.append(full_TAME_runtimes[i,j]/(LREA_runtime+klau_setup_rt+klau_rt))
            


    #
    #  Plot the Data
    #

 
    #  --  Table Data  --  #

    tri_match_idx = 0 
    edge_match_idx = 1
    runtime_idx = 2


    results_to_plot = [
        (LT_tabu_tri_match_ratios,LT_tabu_edge_match_ratios,LT_tabu_runtimes,LT_Tabu_color,LT_Tabu_linestyle,r"$\Lambda$"+"-TAME\nLocal Search",(-.45,-.01)),
        (LT_klau_tri_match_ratios,LT_klau_edge_match_ratios,LT_klau_runtimes,LT_Klau_color,LT_Klau_linestyle,r"$\Lambda$"+"-TAME\nKlau",(-.425,-.01)),
        (LRT_tabu_tri_match_ratios,LRT_tabu_edge_match_ratios,LRT_tabu_runtimes,LRT_Tabu_color,LRT_Tabu_linestyle,"LR-TAME\nLocal Search",(-.475,-.01)),
        #(LRT_lrm_tabu_tri_match_ratios,LRT_lrm_tabu_edge_match_ratios,LRT_lrm_tabu_runtimes,LRT_Tabu_color,LRT_Tabu_linestyle,"LRT-lrm-Tabu",(-.675,-.01)),
        (LRT_klau_tri_match_ratios,LRT_klau_edge_match_ratios,LRT_klau_runtimes,LRT_Klau_color,LRT_Klau_linestyle,"LR-TAME\nKlau",(-.475,-.01)),
        #(LRT_lrm_klau_tri_match_ratios,LRT_lrm_klau_edge_match_ratios,LRT_lrm_klau_runtimes,LRT_Klau_color,LRT_Klau_linestyle,"LRT-lrm-Klau",(-.675,-.01)),
        (LREA_tabu_tri_match_ratios,LREA_tabu_edge_match_ratios,LREA_tabu_runtimes,LREA_Tabu_color,LREA_Tabu_linestyle,"LR-EigenAlign\nLocal Search",(-.55,-.01)),
        (LREA_klau_tri_match_ratios,LREA_klau_edge_match_ratios,LREA_klau_runtimes,LREA_Klau_color,LREA_Klau_linestyle,"LR-EigenAlign\nKlau",(-.55,-.01))
    ]

   
    main_gs = fig.add_gridspec(2, 1,
                               hspace=0.0,wspace=0.2, height_ratios = [1,.25],
                               left=0.2, right=0.975, top=.85, bottom=.05)

    #legend_gs = main_gs[1].subgridspec(nrows=1,ncols=20)


    table_gs = main_gs[0].subgridspec(nrows=len(results_to_plot),ncols=3,wspace=0.05,hspace=0.0)
    microplot_gs = main_gs[1].subgridspec(nrows=1,ncols=3,wspace=0.2,hspace=0.0)
    
    all_axes = np.empty((len(results_to_plot)+1,3),object)
                                            # +1 for microplots

    loess_frac = .3
    annotations = []
    for i,(tri_match_data,edge_match_data,runtime_data,color,linestyle,label,underline_xs) in enumerate(results_to_plot):
        if i == 0:

            for j in [tri_match_idx,runtime_idx,edge_match_idx]:
                all_axes[i,j] = fig.add_subplot(table_gs[i,j])
                all_axes[-1,j] = fig.add_subplot(microplot_gs[-1,j])

        else:
            for j in [tri_match_idx,runtime_idx,edge_match_idx]:
                all_axes[i,j] = fig.add_subplot(table_gs[i,j],sharex=all_axes[0,j])

            #all_axes[i,tri_match_idx] = fig.add_subplot(table_gs[i,tri_match_idx],sharex=all_axes[0,tri_match_idx])
            #all_axes[i,runtime_idx] = fig.add_subplot(table_gs[i,runtime_idx],sharex=all_axes[0,runtime_idx])
            #all_axes[i,edge_match_idx] = fig.add_subplot(table_gs[i,edge_match_idx],sharex=all_axes[0,edge_match_idx])

        make_violin_plot(all_axes[i,tri_match_idx],tri_match_data,precision=2,c=color,v_alpha=.3,format="default",xlim=None,xscale="linear",column_type=None)
        make_violin_plot(all_axes[i,edge_match_idx],edge_match_data,precision=2,c=color,v_alpha=.3,format="default",xlim=None,xscale="log",column_type=None)
        make_violin_plot(all_axes[i,runtime_idx],runtime_data,precision=1,c=color,v_alpha=.3,format="default",xlim=None,xscale="log",column_type=None)
        """
        (line_x_start,line_x_end) = underline_xs
        y = .4
        line_x_start = -.7
        line_x_end = -.01
        all_axes[i,0].annotate('', xy=(line_x_start, y), xycoords='axes fraction', xytext=(line_x_end,y),
                                                arrowprops=dict(arrowstyle="-", color=color,linestyle=linestyle,linewidth=3))
        """
        annotations.append(all_axes[i,0].annotate(label,xy=(-.4,.5), xycoords='axes fraction',ha="center",va="center",c=color,fontsize=9))
        
        plot_1d_loess_smoothing(motif_products,tri_match_data,loess_frac,all_axes[-1,tri_match_idx],c=color,logFilter=True,logFilterAx="x")#,linestyle=linestyle
        plot_1d_loess_smoothing(motif_products,edge_match_data,loess_frac,all_axes[-1,edge_match_idx],c=color,logFilter=True,logFilterAx="x")#,linestyle=linestyle
        plot_1d_loess_smoothing(motif_products,runtime_data,loess_frac,all_axes[-1,runtime_idx],c=color,logFilter=True,logFilterAx="x")#,linestyle=linestyle

        

    #
    #  Format the Axes
    #

    microplot_legend = all_axes[-1,0].inset_axes([-.6,.2,.45,.65])
    violinplot_legend = all_axes[0,0].inset_axes([-.65,1.2,.6,1.0])  
    


    for ax in chain(all_axes.reshape(-1),[microplot_legend,violinplot_legend]):
        #ax.set_xscale("log")
        #ax.grid(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])
        

    make_microplot_legend(microplot_legend)
    make_violin_plot_legend(violinplot_legend)


    for (j,ax) in enumerate(all_axes[-1,:]):
        ax.grid(True)

        ax.set_xscale("log")
        if j != 0:
            ax.set_yscale("log")
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())       
        ax.set_xticks([1e6,1e7,1e8,1e9,1e10,1e11])
        ax.set_xticklabels([])
    bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1,pad=0.01)
    all_axes[-1,0].annotate("linear",xy=(.125,.825),xycoords="axes fraction",ha="center",va="center",rotation=90,fontsize=6).set_bbox(bbox)
            #denotes microplot in column 0 is linear 

    #all_axes[-1,-1].set_yscale("log")

    bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1,pad=0.0)
    #ax.annotate("log",xy=(.1,.4),xycoords="axes fraction",ha="center",rotation=90,fontsize=6).set_bbox(bbox)
    #all_axes[-1,edge_match_idx].set_yscale("log")
    #all_axes[-1,runtime_idx].set_yscale("log")
    all_axes[-1,tri_match_idx].set_yticks(np.linspace(.2,1.8,5))
    all_axes[-1,tri_match_idx].set_ylim(0,2)
    all_axes[-1,edge_match_idx].set_yticks(np.linspace(1.0,7,5))
    all_axes[-1,edge_match_idx].set_ylim(.8,7.5)
    all_axes[-1,runtime_idx].set_yticks([1e0,1e1,1e2,1e3])
    all_axes[-1,runtime_idx].set_ylim(.5,1.5e3)




    #  - row/col labels  -  # 
    title_ypos = 1.2
    line_offset = .3
    line_x_start = .15
    line_x_end = .85
    all_axes[0,tri_match_idx].annotate("Algo. Tri Match\n TAME Tri Match",xy=(.5, title_ypos), xycoords='axes fraction',ha="center")
    all_axes[0,tri_match_idx].annotate('', xy=(.075, title_ypos+line_offset), xycoords='axes fraction', xytext=(.925, title_ypos+line_offset),
                                                arrowprops=dict(arrowstyle="-", color='k'))
    all_axes[0,edge_match_idx].annotate("Algo. Edge Match\n TAME Edge Match",xy=(.5, title_ypos), xycoords='axes fraction',ha="center")
    all_axes[0,edge_match_idx].annotate('', xy=(.0, title_ypos+line_offset), xycoords='axes fraction', xytext=(1.0, title_ypos+line_offset),
                                                arrowprops=dict(arrowstyle="-", color='k'))



    all_axes[0,runtime_idx].annotate("TAME Runtime\n Algo. Runtime",xy=(.5, title_ypos), xycoords='axes fraction',ha="center")
    all_axes[0,runtime_idx].annotate('', xy=(.1, title_ypos+line_offset), xycoords='axes fraction', xytext=(.925, title_ypos+line_offset),
                                                arrowprops=dict(arrowstyle="-", color='k'))
    """
    renderer = fig.canvas.get_renderer()
    for i,row_label in enumerate(annotations):
        print(i)
        underline_text(all_axes[i,0],renderer,row_label,color,linestyle)
    """

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def make_LVGNA_TTVMatchingRatio_runtime_plots(ax=None,useLowRank=False,save_path=None):
    """
        new version of MatchingRatio plot in the new simax-v2 style
    """
    Opt_tri_ratio,Opt_edge_match,\
        Opt_pp_edge_match,Opt_pp_tri_match,\
        Runtimes, TAME_impTTV_runtimes, TAME_BM_runtimes,\
        post_processing_runtime, exp_idx = CT_LP.parse_results()
    graph_names = [" ".join(file.split(".smat")[0].split("_")) for (file,idx) in sorted(exp_idx.items(),key=lambda x:x[1])]

    TAME_BM_ratio = np.divide(TAME_BM_runtimes,TAME_impTTV_runtimes)

    datapath = TAME_RESULTS + "LVGNA_alignments/"

    def process_LowRankTAME_data(f):
        _, results = json.load(f)

        #
        #  Parse Input files
        #

        #Format the data to 
        #exp_idx = {name:i for i,name in enumerate(graph_names)}
        
        matchingRuntimes = np.zeros((len(exp_idx),len(exp_idx)))
        contractionRuntimes = np.zeros((len(exp_idx),len(exp_idx)))

        for (file_A,file_B,matched_tris,max_tris,param_profiles) in results:
            graph_A = file_A.split(".ssten")[0] + ".smat"
            graph_B = file_B.split(".ssten")[0] + ".smat"
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]


            #sum over all params
            totalContractionRuntime = 0.0
            contraction_times = ['qr_timings', 'contraction_timings','svd_timings']
            totalMatchingRuntime = 0.0

            for params, profile in param_profiles:

                contraction_timings = [v for k,v in profile.items() if k in contraction_times]
                totalContractionRuntime += sum(reduce(lambda l1,l2: [x + y for x,y in zip(l1,l2)],contraction_timings))
                totalMatchingRuntime += sum(profile['matching_timings'])
            
            contractionRuntimes[i,j] = totalContractionRuntime
            contractionRuntimes[j,i] = totalContractionRuntime
            matchingRuntimes[i,j] = totalMatchingRuntime
            matchingRuntimes[j,i] = totalMatchingRuntime

        return matchingRuntimes, contractionRuntimes

    with open(datapath+"LowRankTAME_LVGNA_alpha_[.5,1.0]_beta_[0,1e0,1e1,1e2]_iter_15.json", "r") as f:
        matchingRuntimes, contractionRuntimes = process_LowRankTAME_data(f)
        LowRankTAME_ratio = np.divide(matchingRuntimes,contractionRuntimes)
       
        
    with open(datapath+"LowRankTAME_LVGNA_alpha_[.5,1.0]_beta_[0,1,1e1,1e2]_iter_15_low_rank_matching.json", "r") as f:
        matchingRuntimes,contractionRuntimes = process_LowRankTAME_data(f)
        LowRankTAME_LRM_ratio = np.divide(matchingRuntimes,contractionRuntimes)
        

    def parseLambdaTAMEData(exp_results):

        #exp_idx = {name:i for i,name in enumerate(graph_names)}
        ratio_data = np.zeros((len(exp_idx),len(exp_idx)))

        for (file_A,file_B,matched_tris,max_tris,_,runtime) in exp_results:
            graph_A = file_A.split(".ssten")[0] + ".smat"
            graph_B = file_B.split(".ssten")[0] + ".smat"
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]


            contraction_rt = sum(runtime['TAME_timings'])
            matching_rt = sum(runtime['Matching Timings'])

            ratio_data[j,i] = matching_rt/contraction_rt
            ratio_data[i,j] = matching_rt/contraction_rt

        return ratio_data

    #with open(TAME_RESULTS + "LVGNA_Experiments/LambdaTAME_LVGNA_results_alphas:[.5,1.0]_betas:[0,1e0,1e1,1e2]_iter:15.json","r") as f:
    with open(datapath+"LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:rankOneMatching_profile:true_tol:1e-6_results.json","r") as f:
        LambdaTAME_ratio = parseLambdaTAMEData(json.load(f))


    #with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_profile:true_tol:1e-6_results.json","r") as f:  
    with open(datapath+"LVGNA_pairwiseAlignment_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_MatchingTol:1e-8_profile:true_tol:1e-6_results.json","r") as f: 
        LambdaTAMEGramMatching_ratio = parseLambdaTAMEData(json.load(f))


    n = len(graph_names)
    problem_sizes = []

    TAME_exp_ratios = []
    LambdaTAME_exp_ratios = []
    LambdaTAMEGramMatching_exp_ratios = []
    LowRankTAME_exp_ratios = []
    LowRankTAME_LRM_exp_ratios = []

    Is,Js = np.triu_indices(n,k=1)
    for i,j in zip(Is,Js):

        TAME_exp_ratios.append(TAME_BM_ratio[i,j])
        LambdaTAME_exp_ratios.append(LambdaTAME_ratio[i,j])
        LambdaTAMEGramMatching_exp_ratios.append(LambdaTAMEGramMatching_ratio[i,j])
        LowRankTAME_exp_ratios.append(LowRankTAME_ratio[i, j])
        LowRankTAME_LRM_exp_ratios.append(LowRankTAME_LRM_ratio[i, j])
        problem_sizes.append(triangle_counts[graph_names[i]]*triangle_counts[graph_names[j]])


    #
    #  Plot results
    #


    if ax is None:
        fig =plt.figure(figsize=(4.25,3.5))
        spec = fig.add_gridspec(nrows=1, ncols=1,left=.175,right=.9,top=.95,bottom=.125)
        ax = fig.add_subplot(spec[0,0])
        show_plot = True
    else:
        show_plot = False
    #ax = [ax] #jerry rigged

    if useLowRank:
        ax.set_ylim(1e-3,1e5)
        fontsize=10
    else:
        ax.set_ylim(1e-2,1e5)
        fontsize=14
    ax.set_xlim(2e5,5e11)

    label_font_size = None

    #if show_plot:
    ax.set_ylabel("matching runtime\ncontraction runtime",fontsize=label_font_size)
    x_loc = -.19
    ax.annotate('', xy=(x_loc, .29), xycoords='axes fraction', xytext=(x_loc, 0.71),
                arrowprops=dict(arrowstyle="-", color='k'))
    #else:
    #    ax.set_ylabel("matching runtime / contraction runtime")

    ax.set_xlabel(r"|$T_A$||$T_B$|")
    #left axis labels
  
    ax.set_xscale("log")
    ax.set_yscale("log")
    loess_smoothing_frac = .3
    ax.grid(which="major",zorder=-2)
    ax.axhspan(1e-5,1,alpha=.1,color="k")
    
    scatter_alpha = 0.5
    #TODO: update marker types to use global vars

    ax.scatter(problem_sizes,TAME_exp_ratios,label="TAME", c=T_color,marker=T_marker,alpha=scatter_alpha)
    plot_1d_loess_smoothing(problem_sizes,TAME_exp_ratios,loess_smoothing_frac,ax,c=T_color,logFilter=True)#linestyle=T_linestyle,
    #ax[0].plot(range(len(old_TAME_performance)), old_TAME_performance, label="TAME", c=t4_color)
    if useLowRank:
        T_annotationloc = (.1, .55)
    else:
        T_annotationloc = (.5, .05)
    ax.annotate("TAME (C++)",xy=T_annotationloc, xycoords='axes fraction', c=T_color,fontsize=fontsize)
 
    ax.scatter(problem_sizes,LowRankTAME_exp_ratios,label="LowRankTAME", c=LRT_color,alpha=scatter_alpha,marker=LRT_marker)
    plot_1d_loess_smoothing(problem_sizes,LowRankTAME_exp_ratios,loess_smoothing_frac,ax,c=LRT_color,logFilter=True)#linestyle=LRT_linestyle,
    if useLowRank:
        LRT_annotationloc = (.01, .38)
    else:
        LRT_annotationloc = (.8, .36)
    ax.annotate("LowRank\nTAME",xy=LRT_annotationloc, xycoords='axes fraction', c=LRT_color,ha="left",fontsize=fontsize)

  
    ax.scatter(problem_sizes,LambdaTAMEGramMatching_exp_ratios,label="$\Lambda$-TAME", c=LT_color ,marker=LT_marker,alpha=scatter_alpha)
    plot_1d_loess_smoothing(problem_sizes,LambdaTAMEGramMatching_exp_ratios,loess_smoothing_frac,ax,c=LT_color)#,linestyle=LT_linestyle
    #ax[0].plot(range(len(new_TAME_performance)), new_TAME_performance, label="$\Lambda$-TAME", c=t2_color)
    if useLowRank:
        LT_annotationloc = (.1, .825)
    else:
        LT_annotationloc = (.1, .86)
    ax.annotate("$\Lambda$-TAME",xy=LT_annotationloc, xycoords='axes fraction', c=LT_color,fontsize=fontsize)


     

    if useLowRank:
        ax.scatter(problem_sizes,LowRankTAME_LRM_exp_ratios,facecolors='none',edgecolors=LRT_lrm_color,label="LowRankTAME-(lrm)",marker=LRT_lrm_marker,alpha=scatter_alpha)
        plot_1d_loess_smoothing(problem_sizes,LowRankTAME_LRM_exp_ratios,loess_smoothing_frac,ax,c=LRT_lrm_color,logFilter=True)#,linestyle=LRT_lrm_linestyle
        ax.annotate("LowRankTAME-(lrm)",xy=(.075, .05), xycoords='axes fraction', c=LRT_lrm_color,fontsize=fontsize)
        
        #print(new_TAME_exp_runtimes)
        ax.scatter(problem_sizes,LambdaTAME_exp_ratios,label="$\Lambda$-TAME-(rom)",facecolors='none', edgecolors=LT_rom_color,marker=LT_rom_marker,alpha=scatter_alpha)
        plot_1d_loess_smoothing(problem_sizes,LambdaTAME_exp_ratios,loess_smoothing_frac,ax,c=LT_rom_color,logFilter=True)#,linestyle=LT_rom_linestyle
        #ax[0].plot(range(len(new_TAME_performance)), new_TAME_performance, label="$\Lambda$-TAME", c=t2_color)
        ax.annotate("$\Lambda$-TAME\n(rom)",xy=(.25, .225), xycoords='axes fraction', ha="center",c=LT_rom_color,fontsize=fontsize)
        
    #ax.text(.95, .6,"contraction\ndominates")#, ha="center",c="k",rotation=90)
    #ax.text(.95, .1,"contraction\ndominates")#, ha="center",c="k",rotation=90)

    if useLowRank:
        ax.annotate("matching\ndominates",xy=(1.05, .6), xycoords='axes fraction', ha="center",c="k",rotation=90,fontsize=label_font_size)
        ax.annotate("contraction\ndominates",xy=(1.05, 0.05), xycoords='axes fraction', ha="center",c="k",rotation=90,fontsize=label_font_size)
    else:
        ax.annotate("matching\ndominates",xy=(1.05, .55), xycoords='axes fraction', ha="center",c="k",rotation=90,fontsize=label_font_size)
        ax.annotate("contraction\ndominates",xy=(1.05, 0.025), xycoords='axes fraction', ha="center",c="k",rotation=90,fontsize=label_font_size)
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis="both",direction="out",which='both', length=0)

    if show_plot and save_path is None:
        plt.show()

    if not (save_path is None):
        plt.savefig(save_path)

# Supplementary File

def LVGNA_pre_and_post_processed(save_path=None):

    #
    #  Load in Data 
    #


    data_path = TAME_RESULTS + "LVGNA_Alignments/"

    file =  "LVGNA_pairwiseAlignment_LambdaTAME_MultiMotif_alphas:[1.0,0.5]_betas:[0.0,1.0,10.0,100.0]_order:3_samples:3000000_data.json"
    file = "LVGNA_pairwiseAlignment_LambdaTAME_alphas:[0.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_postProcessing:KlauAlgo_profile:true_tol:1e-6_results.json"
    with open(data_path +file,"r") as f:
        data = json.load(f)
        vertex_products = []
        edge_products = []
        motif_products = []

        LT_klau_tri_match = []
        LT_klau_edge_match = []
        LT_klau_runtimes = []
        filenames = []

        #for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LT_profile,LT_edges_matched,klau_edges_matched,klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity) in data:
        for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,LT_profile,LT_edges_matched,klau_edges_matched,klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity,f_status) in data:
            
            filenames.append((file_i,file_j))
            vertex_A = vertex_counts[" ".join(file_i.split(".smat")[0].split("_"))]
            vertex_B = vertex_counts[" ".join(file_j.split(".smat")[0].split("_"))]
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]


            LT_klau_runtimes.append(klau_setup_rt+klau_rt)
            vertex_products.append(vertex_A*vertex_B)
            edge_products.append(edges_A*edges_B)
            #motif_products.append(A_Motifs[0]*B_Motifs[0])
            motif_products.append(A_Motifs*B_Motifs)

            LT_klau_edge_match.append(klau_edges_matched/min(edges_A,edges_B))
            #LT_klau_tri_match.append(klau_tris_matched/min(A_Motifs[0],B_Motifs[0]))
            LT_klau_tri_match.append(klau_tris_matched/min(A_Motifs,B_Motifs))

    """
    with open(data_path + "LVGNA_pairwiseAlignment_LambdaTAME_MultiMotif_alphas:[1.0,0.5]_betas:[0.0,1.0,10.0,100.0]_order:3_postProcessing:SuccessiveKlau-const-iter:5-maxIter:500_samples:10000000_data.json","r") as f:
        data = json.load(f)
        successive_klau_tri_match_ratios = []
        successive_klau_edge_match_ratios = []
        successive_klau_runtimes = []
     


        for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LT_profile,LT_edges_matched,sklau_edges_matched,sklau_tris_matched,_,successive_klau_profiling) in data:
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            
            successive_klau_edge_match_ratios.append(sklau_edges_matched/min(edges_A,edges_B))
            successive_klau_tri_match_ratios.append(sklau_tris_matched/min(A_Motifs[0],B_Motifs[0]))
            successive_klau_runtimes.append(sum(successive_klau_profiling["runtime"]))
    """
    #LT data comes from here 
    file = "LVGNA_pairwiseAlignment_LambdaTAME_MultiMotif_alphas:[1.0,0.5]_betas:[0.0,1.0,10.0,100.0]_order:3_postProcessing:TabuSearch_samples:30000000_data.json"
    file = "LVGNA_pairwiseAlignment_LambdaTAME_alphas:[0.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_postProcessing:LocalSearch_profile:true_tol:1e-6_results.json"
    
    with open(data_path + file,"r") as f:
        data = json.load(f)
        LT_tabu_tri_match = []
        LT_tabu_edge_match = []
        LT_tabu_runtimes = []
        LT_tri_match = []
        LT_runtimes = []

        #for i,(file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LT_profile,LT_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,full_rt) in enumerate(data):
        for i,(file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,LT_profile,LT_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,full_rt) in enumerate(data):
            assert (file_i,file_j) == filenames[i]
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            
            #LT_tri_match.append(LT_matchedMotifs[0]/min(A_Motifs[0],B_Motifs[0]))
            LT_tri_match.append(LT_matchedMotifs/min(A_Motifs,B_Motifs))
            LT_runtimes.append(sum([sum(profile) for profile in LT_profile.values()]))


            LT_tabu_edge_match.append(tabu_edges_matched/min(edges_A,edges_B))
            #LT_tabu_tri_match.append(tabu_tris_matched/min(A_Motifs[0],B_Motifs[0]))
            LT_tabu_tri_match.append(tabu_tris_matched/min(A_Motifs,B_Motifs))
            LT_tabu_runtimes.append(full_rt)

    def parse_Cpp_data(files):
        # given the file pairs to make the data consistent with 

        preP_tri_match_ratio = []
        preP_edge_match_ratio = []
        preP_runtime = []

        postP_tri_match_ratio = []
        postP_edge_match_ratio = []
        postP_runtime = []

        Opt_tri_ratio,Opt_edge_match,\
        Opt_pp_edge_match,Opt_pp_tri_match,\
        Runtimes, impTTV_runtimes, matching_runtimes,\
        post_processing_runtime, indexing = CT_LP.parse_results()

        for file_idx,(file_i,file_j) in enumerate(files):
            i = indexing[file_i]
            j = indexing[file_j]

            preP_tri_match_ratio.append(Opt_tri_ratio[i,j])    
            preP_edge_match_ratio.append(Opt_edge_match[i,j])
            preP_runtime.append(Runtimes[i,j])

            postP_tri_match_ratio.append(Opt_pp_tri_match[i,j])    
            postP_edge_match_ratio.append(Opt_pp_edge_match[i,j])
            postP_runtime.append(post_processing_runtime[i,j])

        return preP_tri_match_ratio, preP_edge_match_ratio, preP_runtime, postP_tri_match_ratio, postP_edge_match_ratio, postP_runtime, indexing

    # LREA data comes from here 
    with open(data_path + "LVGNA_pairwiseAlignment_LowRankEigenAlign_postProcessing:TabuSearch_profile:true_results.json") as f:
        data = json.load(f)
        #for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,LREA_profile,LREA_edges_matched,LREA_klau_edges_matched,LREA_klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity,f_status) in data:
        
        LREA_tri_match = []
        LREA_runtimes = []
        LREA_tabu_tri_match = []
        LREA_tabu_edge_match = []
        LREA_tabu_runtimes = []
        for i,(file_i,file_j,LREA_matchedMotifs,A_Motifs,B_Motifs,LREA_profile,LREA_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,full_rt) in enumerate(data):
            
            assert filenames[i] == (file_i,file_j)
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            

            LREA_tri_match.append(LREA_matchedMotifs/min(A_Motifs,B_Motifs))
            LREA_runtimes.append(LREA_profile)
            LREA_tabu_edge_match.append(tabu_edges_matched/min(edges_A,edges_B))
            LREA_tabu_tri_match.append(tabu_tris_matched/min(A_Motifs,B_Motifs))
            LREA_tabu_runtimes.append(full_rt)

    with open(data_path + "LVGNA_pairwiseAlignment_LowRankEigenAlign_postProcessing:KlauAlgo_profile:true_results.json") as f:
        data = json.load(f)
        
        LREA_klau_tri_match = []
        LREA_klau_edge_match = []
        LREA_klau_runtimes = []
    
        for i,(file_i,file_j,_,A_Motifs,B_Motifs,LREA_profile,LREA_edges_matched,LREA_klau_edges_matched,LREA_klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity,f_status) in enumerate(data):
            assert filenames[i] == (file_i,file_j)
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            
            LREA_klau_edge_match.append(LREA_klau_edges_matched/min(edges_A,edges_B))
            LREA_klau_tri_match.append(LREA_klau_tris_matched/min(A_Motifs,B_Motifs))
            LREA_klau_runtimes.append(klau_setup_rt+klau_rt)


    file = "LVGNA_pairwiseAlignment_LowRankTAME_alphas:[0.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_order:3_postProcessing:KlauAlgo_profile:true_samples:3000000_tol:1e-6_results.json"
    file = "LVGNA_pairwiseAlignment_LowRankTAME_alphas:[0.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_postProcessing:KlauAlgo_profile:true_tol:1e-6_results_colMajor.json"
    with open(data_path + file,"r") as f:
        data = json.load(f)
        #add in missing \beta = 1.0 case
  

        LRT_klau_tri_match = []
        LRT_klau_edge_match = []
        LRT_klau_runtimes = []

        #for i,(file_i,file_j,LRT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LRT_profile,LRT_edges_matched,klau_edges_matched,klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity,f_status) in enumerate(data):
        for i,(file_i,file_j,LRT_matchedMotifs,A_Motifs,B_Motifs,LRT_profile,LRT_edges_matched,klau_edges_matched,klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity,f_status) in enumerate(data):
            assert filenames[i] == (file_i,file_j)
            #i = gn_dict[" ".join(file_i.split(".smat")[0].split("_"))]
            #j = gn_dict[" ".join(file_j.split(".smat")[0].split("_"))]
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            

            LRT_klau_edge_match.append(klau_edges_matched/min(edges_A,edges_B))
            LRT_klau_tri_match.append(klau_tris_matched/min(A_Motifs,B_Motifs))
            LRT_klau_runtimes.append(klau_setup_rt + klau_rt)
    
    # LRT data comes from here 
    file = "LVGNA_pairwiseAlignment_LowRankTAME_alphas:[1.0,0.5]_betas:[0.0,1.0,10.0,100.0]_iter:15_order:3_postProcessing:TabuSearch_profile:true_tol:1e-6_results.json"
    file = "LVGNA_pairwiseAlignment_LowRankTAME_alphas:[0.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_postProcessing:LocalSearch_profile:true_tol:1e-6_results_colMajor.json"
    with open(data_path + file,"r") as f:
        data = json.load(f)


        LRT_tri_match = []
        LRT_runtimes = []
        LRT_tabu_tri_match = []
        LRT_tabu_edge_match = []
        LRT_tabu_runtimes = []
        LRT_runtimes = []

        #for i,(file_i,file_j,LRT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LRT_profile,LRT_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,tabu_full_rt)  in enumerate(data):
        for i,(file_i,file_j,LRT_matchedMotifs,A_Motifs,B_Motifs,LRT_profile,LRT_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,tabu_full_rt)  in enumerate(data):
            assert filenames[i] == (file_i,file_j)
            #i = gn_dict[" ".join(file_i.split(".smat")[0].split("_"))]
            #j = gn_dict[" ".join(file_j.split(".smat")[0].split("_"))]
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            
            LRT_tabu_edge_match.append(tabu_edges_matched/min(edges_A,edges_B))
            LRT_tabu_tri_match.append(tabu_tris_matched/min(A_Motifs,B_Motifs))
            LRT_tabu_runtimes.append(tabu_full_rt)

            pre_processed_time = 0.0
            for (param, profile) in LRT_profile:
                pre_processed_time += sum([sum(v) for (k,v) in profile.items() if "timings" in k])
            LRT_tri_match.append(LRT_matchedMotifs/min(A_Motifs,B_Motifs))
            LRT_runtimes.append(pre_processed_time)


    TAME_preP_tri_match_ratio, TAME_preP_edge_match_ratio, TAME_preP_runtime,\
    TAME_postP_tri_match_ratio, TAME_postP_edge_match_ratio, TAME_postP_runtime, \
    exp_indexing = parse_Cpp_data(filenames)
    
    graph_names = [" ".join(file.split(".smat")[0].split("_")) for (file,idx) in sorted(exp_indexing.items(),key=lambda x:x[1])]
    gn_dict = {graph:i for (i,graph) in enumerate(graph_names)}
    
    LGRAAL_graphs, LGRAAL_tri_results, LGRAAL_runtimes, _,_, _,_ = get_results()
    LGRAAL_perm = [gn_dict[graph] for graph in LGRAAL_graphs]

    LGRAAL_tri_results = LGRAAL_tri_results[np.ix_(LGRAAL_perm, LGRAAL_perm)]
    LGRAAL_runtimes = LGRAAL_runtimes[np.ix_(LGRAAL_perm, LGRAAL_perm)]

    LGRAAL_tri_match = []
    LGRAAL_runtime = []
    for (file_i,file_j) in filenames: 
        i = exp_indexing[file_i]
        j = exp_indexing[file_j]

        LGRAAL_tri_match.append(LGRAAL_tri_results[i,j])
        LGRAAL_runtime.append(LGRAAL_runtimes[i,j])
    


    #
    #  Plot the Data
    # 

    fig = plt.figure(figsize=(9,7))
    
    gs = fig.add_gridspec(nrows=2, ncols=1,
                          left=0.05, right=0.975,top=.95,bottom=.1,
                          wspace=0.1,hspace=0.1
                          )
    height_ratios = [.5,1.0,1.0]
    tri_match_gs = gs[0].subgridspec(nrows=3,ncols=5,hspace=0.4,wspace=0.2,height_ratios=height_ratios)
    tri_match_axes = np.empty((3,5),object)
    
    runtime_gs = gs[1].subgridspec(nrows=1,ncols=2,
                                #left=0.125, right=0.9,top=.95,bottom=.075,
                                wspace=0.1,hspace=0.1)
    runtime_axes = np.empty(2,object)

    #  --  Plot Runtime Data  --  # 
    for i in range(2):
        runtime_axes[i] = fig.add_subplot(runtime_gs[i])
    pre_index = 0
    post_index = 1
    loess_frac = .3
    annotation_fontsize = 10


    preP_data_to_plot = [
        (LRT_runtimes,LRT_color,LRT_linestyle,"LowRankTAME",(.665,.61),12.6,"both"),#-.05,.43)
        (LREA_runtimes,LREigenAlign_color,LREigenAlign_linestyle,"LowRankEigenAlign",(.5,.075),7.5,"both"),
        (LT_runtimes,LT_color,LT_linestyle,r"$\Lambda-$TAME",(.35,.3),0,"both"),
        (TAME_preP_runtime,T_color,T_linestyle,"TAME (C++)",(.6,.7),20,"both"),
        (LGRAAL_runtime,LGRAAL_color,LGRAAL_linestyle,"LGRAAL",(.3,.6),0,"both"),
    ]


    for (tri_match_data,color,linestyle,annotation_text,annotation_loc,annotation_angle,logFilterAx) in preP_data_to_plot:
        #runtime_axes[pre_index].scatter(motif_products,tri_match_data)
        plot_1d_loess_smoothing(motif_products,tri_match_data,loess_frac,runtime_axes[pre_index],
                                c=color,logFilter=True,logFilterAx=logFilterAx)#linestyle=linestyle,
        runtime_axes[pre_index].annotate(annotation_text,xy=annotation_loc,c=color,
                                         xycoords='axes fraction',ha="left",rotation=annotation_angle,
                                         fontsize=annotation_fontsize)
 

    postP_data_to_plot = [
        (LRT_klau_runtimes,LRT_Klau_color,LRT_Klau_linestyle,"LRT-Klau",(.1,.45),0),
        (LRT_tabu_runtimes,LRT_Tabu_color,LRT_Tabu_linestyle,"LRT-LS",(.1,.175),-5),
        (LREA_klau_runtimes,LREA_Klau_color,LRT_Klau_linestyle,"LREA-Klau",(.35,.41),12.5),
        (LREA_tabu_runtimes,LREA_Tabu_color,LREA_Tabu_linestyle,"LREA-LS",(.9,.6),12.5),
        (LT_klau_runtimes,LT_Klau_color,LT_Klau_linestyle,r"$\Lambda$T-Klau",(.9,.45),10),
        (LT_tabu_runtimes,LT_Tabu_color,LT_Tabu_linestyle,r"$\Lambda$T-LS",(.525,.225),7.5),
        (TAME_postP_runtime,T_color,T_linestyle,"TAME (C++)\nLocal Search",(.525,.75),0),
    ]

    for (tri_match_data,color,linestyle,annotation_text,annotation_loc,annotation_angle) in postP_data_to_plot:
        plot_1d_loess_smoothing(motif_products,tri_match_data,loess_frac,runtime_axes[post_index],
                                c=color,logFilter=True,logFilterAx="both")#linestyle=linestyle,
        runtime_axes[post_index].annotate(annotation_text,xy=annotation_loc,c=color,
                                         xycoords='axes fraction',ha="center",rotation=annotation_angle,
                                         fontsize=annotation_fontsize)

    annotation_fontsize = 12
    #  --  Plot Tri Match Data  --  # 
    for i in range(3):
        for j in range(5):
            tri_match_axes[i,j] = fig.add_subplot(tri_match_gs[i,j])

    preP_triMatch_data_to_plot = [
        (LGRAAL_tri_match,LGRAAL_color,LGRAAL_linestyle),
        (LT_tri_match,LT_color,LT_linestyle),
        (LRT_tri_match,LRT_color,LRT_linestyle),
        (LREA_tri_match,LREigenAlign_color,LREigenAlign_linestyle),
        (TAME_preP_tri_match_ratio,T_color,T_linestyle),
    ]

    for j,(exp_tri_match,color,linestyle) in enumerate(preP_triMatch_data_to_plot):
        tri_match_axes[0,j].scatter(motif_products,exp_tri_match,s=5,c=color,zorder=3,alpha=.5)
        plot_1d_loess_smoothing(motif_products,exp_tri_match,loess_frac,tri_match_axes[0,j],
                                c=color,logFilter=True,logFilterAx="x")#,linestyle=linestyle

    postP_triMatch_data_to_plot = [
        [(None,LGRAAL_color,LGRAAL_linestyle)],
        [(LT_tabu_tri_match,LT_Tabu_color,LT_Tabu_linestyle),(LT_klau_tri_match,LT_Klau_color,LT_Klau_linestyle)],
        [(LRT_tabu_tri_match,LRT_Tabu_color,LRT_Tabu_linestyle),(LRT_klau_tri_match,LRT_Klau_color,LRT_Klau_linestyle)],
        [(LREA_tabu_tri_match,LREA_Tabu_color,LREA_Tabu_linestyle),(LREA_klau_tri_match,LREA_Klau_color,LREA_Klau_linestyle)],
        [(TAME_postP_tri_match_ratio,T_color,T_linestyle)],
    ]

    for j,post_processed_exps in enumerate(postP_triMatch_data_to_plot):
        for (exp_tri_match,color,linestyle) in post_processed_exps:
            if exp_tri_match is not None:
                tri_match_axes[1,j].scatter(motif_products,exp_tri_match,s=5,c=color,zorder=3,alpha=.25)
                plot_1d_loess_smoothing(motif_products,exp_tri_match,loess_frac,tri_match_axes[1,j],
                                        c=color,logFilter=True,logFilterAx="both")#linestyle=linestyle,

    postP_edgeMatch_data_to_plot = [
        [(None,LGRAAL_color,LGRAAL_linestyle)],
        [(LT_tabu_edge_match,LT_Tabu_color,LT_Tabu_linestyle),(LT_klau_edge_match,LT_Klau_color,LT_Klau_linestyle)],
        [(LRT_tabu_edge_match,LRT_Tabu_color,LRT_Tabu_linestyle),(LRT_klau_edge_match,LRT_Klau_color,LRT_Klau_linestyle)],
        [(LREA_tabu_edge_match,LREA_Tabu_color,LREA_Tabu_linestyle),(LREA_klau_edge_match,LREA_Klau_color,LREA_Klau_linestyle)],
        [(TAME_postP_edge_match_ratio,T_color,T_linestyle)],
    ]

    for j,post_processed_exps in enumerate(postP_edgeMatch_data_to_plot):
        for (exp_tri_match,color,linestyle) in post_processed_exps:
            if exp_tri_match is not None:
                tri_match_axes[2,j].scatter(motif_products,exp_tri_match,s=5,c=color,zorder=3,alpha=.25)
                plot_1d_loess_smoothing(motif_products,exp_tri_match,loess_frac,tri_match_axes[2,j],
                                        c=color,logFilter=True,logFilterAx="both")#linestyle=linestyle,


    #
    #   Touch up the Axes
    #

    for ax in chain(runtime_axes.reshape(-1),tri_match_axes.reshape(-1)):
        ax.set_xscale("log")
        ax.grid(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)
        ax.set_xticks([1e6,1e7,1e8,1e9,1e10,1e11])  

    for ax in runtime_axes:
        ax.set_xlabel(r"$|T_A||T_B|$")
        ax.set_yscale("log")
        ax.set_ylim(5e0,3e6)
        ax.set_yticks([1e1,1e2,1e3,1e4,1e5,1e6])  
        ax.tick_params(axis="y",which="both",pad=6)    
    runtime_axes[0].set_yticklabels([])

    for j,ax in enumerate(tri_match_axes[0,:]):
        ax.set_ylim(0,.5)
        ax.set_yticks([0.0,.25,.5]) 
        if j == 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(["0",".25",".5"],ha='center')

    for j,ax in enumerate(tri_match_axes[1,:]):
        ax.set_ylim(0,1)
        ax.set_yticks([0.0,.25,.5,.75,1.0]) 
        if j == 1:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(["0",".25",".5",".75","1"],ha='center')

    for j,ax in enumerate(tri_match_axes[2,:]):
        ax.set_ylim(0,.75)
        ax.set_yticks([0.0,.25,.5,.75]) 
        if j == 1:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(["0",".25",".5",".75"],ha='center')


    for ax in tri_match_axes.reshape(-1):
        ax.set_xticklabels([])  
        ax.tick_params(axis="y",which="both",pad=12)

    tri_match_axes[1,0].axes.set_axis_off()
    tri_match_axes[2,0].axes.set_axis_off()
            #LGRAAL doesn't have post processing

    preP_tri_match_annotations = [
        ("LGRAAL",LGRAAL_color,(.5,1.3)),
        (r"$\Lambda-$TAME",LT_color,(.5,1.3)),
        ("LowRankTAME",LRT_color,(.5,1.3)),
        ("LowRank\nEigenAlign",LREigenAlign_color,(.5,1.3)),
        ("TAME",T_color,(.5,1.3)),
    ]
    for j,(label,color,xy) in enumerate(preP_tri_match_annotations):
        tri_match_axes[0,j].annotate(label,xy=xy,c=color,xycoords='axes fraction',ha="center",va="top",
                                         fontsize=annotation_fontsize)

    postP_tri_match_annotations = [
        [(None,LGRAAL_color,(.5,.5),1)],
        [(r"$\Lambda$T-Klau",LT_Klau_color,(.8,.01),1),(r"$\Lambda$T-LS",LT_Tabu_color,(.7,.7),1)],
        [("LRT-Klau",LRT_Klau_color,(.75,-.1),1),("LRT-LS",LRT_Tabu_color,(.5,.65),1)],
        [("LREA-Klau",LREA_Klau_color,(.7,-.15),1),("LREA-LS",LREA_Tabu_color,(.65,.65),1)],
        [("TAME (C++)\nLocal Search",T_color,(.4,.375),2)],
    ]
    for j,postP_annotations in enumerate(postP_tri_match_annotations):
        for (label,color,xy,row_idx) in postP_annotations:
            if label is not None:
                tri_match_axes[row_idx,j].annotate(label,xy=xy,c=color,xycoords='axes fraction',ha="center",
                                                fontsize=annotation_fontsize)




    tri_match_axes[0,0].set_ylabel("matched tris\n"+r"$\min{\{|T_A|,|T_B|\}}$",labelpad=-7.5)
    xloc = -.18
    tri_match_axes[0,0].annotate('', xy=(xloc, -.4), xycoords='axes fraction', xytext=(xloc, 1.4),
                arrowprops=dict(arrowstyle="-", color='k'))
    tri_match_axes[0,0].set_xlabel(r"$|T_A||T_B|$")


    tri_match_axes[1,1].set_ylabel("(refined)\nmatched tris\n"+r"$\min{\{|T_A|,|T_B|\}}$",labelpad=-7.5)
    xloc = -.18
    tri_match_axes[1,1].annotate('', xy=(xloc, .025), xycoords='axes fraction', xytext=(xloc, .975),
                arrowprops=dict(arrowstyle="-", color='k'))

    tri_match_axes[2,1].set_ylabel("(refined)\nmatched edges\n"+r"$\min{\{|E_A|,|E_B|\}}$",labelpad=-7.5)
    xloc = -.18
    tri_match_axes[2,1].annotate('', xy=(xloc, -.05), xycoords='axes fraction', xytext=(xloc, 1.05),
                arrowprops=dict(arrowstyle="-", color='k'))

    runtime_axes[0].set_ylabel("runtime (s)")

    title_size = 18
    bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.05)
    x_loc = 0.18
    y_loc = .975
    y_gap = .1
    runtime_axes[0].annotate("Pre",xy=(x_loc,y_loc), xycoords='axes fraction',ha="center",va="top",fontsize=title_size).set_bbox(bbox) 
    runtime_axes[0].annotate("Processed",xy=(x_loc,y_loc-y_gap), xycoords='axes fraction',ha="center",va="top",fontsize=title_size//2).set_bbox(bbox) 
    x_loc = 0.2025
    runtime_axes[1].annotate("Post",xy=(x_loc,y_loc), xycoords='axes fraction',ha="center",va="top",fontsize=title_size).set_bbox(bbox) 
    runtime_axes[1].annotate("Processed",xy=(x_loc,y_loc-y_gap), xycoords='axes fraction',ha="center",va="top",fontsize=title_size//2).set_bbox(bbox) 
    #  --  #

 
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



#
#   Random Graph Experiment Plots 
#

def RandomGeometricRG_PostProcessing_is_needed(save_path=None):

    data_location = TAME_RESULTS + "klauExps/"
    fig = plt.figure(figsize=(4.5,4.5))

    #fig, axes = plt.subplots(2,2,figsize=(4.25,5))
    n = 2
    m = 2
    spec = fig.add_gridspec(nrows=n, ncols=m,hspace=0.1,wspace=0.15,left=.15,right=.975,top=.95,bottom=.15)
    all_ax = []
    axes = np.empty((n,m),object)
    for i in range(n):
        for j in range(m):
            ax = fig.add_subplot(spec[i,j])
            axes[i,j] = ax



    def process_ER_Noise_data(data,version="klau_old"):
        p_idx = {p:i for (i,p) in enumerate(sorted(set([datum[1] for datum in data])))}
        n_idx = {p:i for (i,p) in enumerate(sorted(set([datum[2] for datum in data])))}
        trials = int(len(data)/(len(p_idx)*len(n_idx)))

        accuracy = np.zeros((len(p_idx),len(n_idx),trials))
        LT_klauAccuracy = np.zeros((len(p_idx),len(n_idx),trials))
        triMatch = np.zeros((len(p_idx),len(n_idx),trials))
        LT_klauTriMatch = np.zeros((len(p_idx),len(n_idx),trials))
        trial_idx = np.zeros((len(p_idx),len(n_idx)),int)


        for datum in data:

            if version == "klau_new":
                (seed,p,n,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt,L_sparsity,fbounds) = datum
            elif version == "klau_old":
                (seed,p,n,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt) = datum
            elif version == "tabu":
                (seed,p,n,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt) = datum
            else:
                raise ValueError("only supports ")
            i = p_idx[p]
            j = n_idx[n]

            accuracy[i,j,trial_idx[i,j]] = acc
            LT_klauAccuracy[i,j,trial_idx[i,j]] = klau_acc
            triMatch[i,j,trial_idx[i,j]] = matched_tris/min(tri_A,tri_B)
            LT_klauTriMatch[i,j,trial_idx[i,j]] = klau_tri_match/min(tri_A,tri_B)
            trial_idx[i,j] += 1
        
        return accuracy, LT_klauAccuracy, triMatch, LT_klauTriMatch, p_idx, n_idx 

    def process_Dup_Noise_data(data,version="old"):

        p_idx = {p:i for (i,p) in enumerate(sorted(set([datum[1] for datum in data])))}
        n_idx = {n:i for (i,n) in enumerate(sorted(set([datum[2] for datum in data])))}
        sp_idx = {sp:i for (i,sp) in enumerate(sorted(set([datum[3] for datum in data])))}

        trials = int(len(data)/(len(p_idx)*len(n_idx)*len(sp_idx)))

        accuracy = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        klauAccuracy = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        triMatch = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        LT_klauTriMatch = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        trial_idx = np.zeros((len(p_idx),len(n_idx),len(sp_idx)),int)

        for datum in data:
            
            if version == "old":
                (seed,p,n,sp,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt) = datum
            elif version == "new klau":
                (seed,p,n,sp,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt,L_sparsity,fstatus) = datum
            else:
                print(f"only supports 'old' and 'new klau', got {version}")

            i = p_idx[p]
            j = n_idx[n]
            k = sp_idx[sp]
            accuracy[i,j,k,trial_idx[i,j,k]] = acc
            klauAccuracy[i,j,k,trial_idx[i,j,k]] = klau_acc
            triMatch[i,j,k,trial_idx[i,j,k]] = matched_tris/min(tri_A,tri_B)
            LT_klauTriMatch[i,j,k,trial_idx[i,j,k]] = klau_tri_match/min(tri_A,tri_B)
            trial_idx[i,j,k] += 1
        
        return accuracy, klauAccuracy, triMatch,LT_klauTriMatch, p_idx, n_idx, sp_idx  


    def make_percentile_plot(plot_ax, x_domain,data,color,hatch=None,**kwargs):
        #TODO: check on scope priorities of ax for closures   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
            (20,80,.05,color)
        ]
        
        #plot_percentiles(plot_ax,  data.T, x_domain, lines, ribbons,**kwargs)
        n,m = data.shape
        percentile_linewidth=.01

        for (lower_percentile,upper_percentile,alpha,color) in ribbons:
            #plot_ax.plot(np.percentile(data.T, lower_percentile, axis=0),c=color,linewidth=percentile_linewidth)
            #plot_ax.plot(np.percentile(data.T, upper_percentile, axis=0),c=color,linewidth=percentile_linewidth)

            plot_ax.fill_between(x_domain,
                            np.percentile(data.T, lower_percentile, axis=0),
                            np.percentile(data.T, upper_percentile, axis=0),
                            facecolor=color,alpha=.1,edgecolor=color)

        for (col_func,alpha,color) in lines:
            line_data = []

            for i in range(n):
                line_data.append(col_func(data[i,:]))
        
            plot_ax.plot(x_domain,line_data,alpha=alpha,c=color,**kwargs)

    #
    #  Erdos Reyni Noise 
    #
    data_location = TAME_RESULTS + "RG_ERNoise/"
    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json'
    with open(data_location + file,'r') as f:
        accuracy,  LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx = process_ER_Noise_data(json.load(f),"klau_new")

    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:TabuSearch_trialcount:20.json'
    with open(data_location + file,'r') as f:
        _,  LT_TabuAccuracy, _, LT_TabuTriMatch, p_idx, n_idx = process_ER_Noise_data(json.load(f),"tabu")

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LRT_accuracy,  LRT_KlauAccuracy, LRT_triMatch, LRT_KlauTriMatch, p_idx, n_idx = process_ER_Noise_data(json.load(f),"klau_new")

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _, LRT_TabuAccuracy,_,LRT_TabuTriMatch, p_idx, n_idx = process_ER_Noise_data(json.load(f),"tabu")


    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_lrm_KlauAccuracy, _, LRT_lrm_KlauTriMatch, p_idx, n_idx = process_ER_Noise_data(json.load(f),"klau_new")

    file = "LowRankEigenAlign_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LREA_Accuracy,  LREA_KlauAccuracy, LREA_TriMatch, LREA_KlauTriMatch, p_idx, n_idx = process_ER_Noise_data(json.load(f),"klau_new")

    file = "LowRankEigenAlign_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LREA_TabuAccuracy, _, LREA_TabuTriMatch, _, _ = process_ER_Noise_data(json.load(f),"tabu")


    sub_ax = axes[:,0]
    spectral_embedding_exps = [
        (accuracy,triMatch,LT_color,LT_linestyle),
        (LRT_accuracy,LRT_triMatch,LRT_color,LRT_linestyle),
        (LREA_Accuracy, LREA_TriMatch,LREigenAlign_color,LREigenAlign_linestyle),
    ]


    post_processing_exps = [
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,LT_Tabu_linestyle),
        #(LRT_lrm_KlauAccuracy,LRT_lrm_KlauTriMatch,LRM_lrm_Klau_color),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,LRT_Klau_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,LRT_Tabu_linestyle),
        (LREA_TabuAccuracy, LREA_TabuTriMatch,LREA_Tabu_color,LREA_Tabu_linestyle),
        (LREA_KlauAccuracy, LREA_KlauTriMatch,LREA_Klau_color,LREA_Klau_linestyle),
    ]

    for (acc,tri,c,linestyle) in chain(spectral_embedding_exps,post_processing_exps):
        make_percentile_plot(axes[0,0],n_idx.keys(),acc[0,:,:],c,linestyle=linestyle)
        make_percentile_plot(axes[1,0],n_idx.keys(),tri[0,:,:],c,linestyle=linestyle)

    #
    #  Duplcation Noise 
    #
    default_p = .5
    default_sp = .25

    sub_ax = axes[:,1]

    data_location = TAME_RESULTS + "RG_DupNoise/"
    
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        accuracy,  LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx, sp_idx = process_Dup_Noise_data(json.load(f))
    
    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json' 
    with open(data_location + file,'r') as f:
        _,  LT_TabuAccuracy,_,LT_TabuTriMatch, p_idx, n_idx, sp_idx = process_Dup_Noise_data(json.load(f))
    
    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LRT_accuracy,  LRT_TabuAccuracy,LRT_triMatch,LRT_TabuTriMatch, p_idx, n_idx, sp_idx = process_Dup_Noise_data(json.load(f))
    
    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_KlauAccuracy,_,LRT_KlauTriMatch, p_idx, n_idx, sp_idx = process_Dup_Noise_data(json.load(f),"new klau")
    
    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_lrm_KlauAccuracy,_,LRT_lrm_KlauTriMatch, p_idx, n_idx, sp_idx = process_Dup_Noise_data(json.load(f),"new klau")
    
    file ="LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LREA_Accuracy,  LREA_KlauAccuracy,LREA_TriMatch,LREA_KlauTriMatch, _, _, _ = process_Dup_Noise_data(json.load(f),"new klau")


    file ="LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LREA_TabuAccuracy,_,LREA_TabuTriMatch, _, _, _ = process_Dup_Noise_data(json.load(f))

    n_exps = [
        (accuracy,triMatch,LT_color,None,LT_linestyle),
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,None,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,None,LT_Tabu_linestyle),
        
        (LRT_accuracy,LRT_triMatch,LRT_color,None,LRT_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,None,LRT_Tabu_linestyle),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,None,LRT_Klau_linestyle),

        (LREA_Accuracy, LREA_TriMatch,LREigenAlign_color,None,LREigenAlign_linestyle),
        (LREA_KlauAccuracy, LREA_KlauTriMatch,LREA_Klau_color,None,LREA_Klau_linestyle),
        (LREA_TabuAccuracy,LREA_TabuTriMatch,LREA_Tabu_color,None,LREA_Tabu_linestyle),
        #(LRT_lrm_KlauAccuracy,LRT_lrm_KlauTriMatch,LRT_lrm_Klau_color),
    ]

    for (acc, tri, c,hatch,linestyle) in n_exps:
        make_percentile_plot(sub_ax[0],n_idx.keys(),acc[p_idx[default_p],:,sp_idx[default_sp],:],c,linestyle=linestyle)
        make_percentile_plot(sub_ax[1],n_idx.keys(),tri[p_idx[default_p],:,sp_idx[default_sp],:],c,linestyle=linestyle)


    #
    #   Final touches on axis
    #
    title_size = 12

    bbox = dict(boxstyle="round", fc="w",ec="w",alpha=1.0,pad=.05)
    axes[0,0].annotate(u"Erdős Rényi",xy=(.975,.975), xycoords='axes fraction',ha="right",va="top",fontsize=title_size).set_bbox(bbox)
    axes[0,1].annotate("Duplication",xy=(.975,.975), xycoords='axes fraction',ha="right",va="top",fontsize=title_size).set_bbox(bbox)


    axes[1,1].annotate(r"$\Lambda$T",xy=(.4,.25),color=LT_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-15)
    axes[1,1].annotate("LRT",xy=(.4,.1),color=LRT_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-7.5)
    axes[1,1].annotate("LREA",xy=(.01,.005),color=LREigenAlign_color,xycoords="axes fraction",ha="left",va="bottom",fontsize=10,rotation=0)


    axes[1,0].annotate("LRT-Klau",xy=(.575,.325),color=LRT_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-5)
    axes[1,0].annotate(r"$\Lambda$T-"+"Klau",xy=(.6,.62),color=LT_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-26)
    axes[1,0].annotate("LREA-Klau",xy=(.275,.55),color=LREA_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-35)
    

    axes[0,1].annotate(r"$\Lambda$T-"+"LS",xy=(.775,.525),color=LT_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-2.5)
    axes[0,1].annotate("LRT-LS",xy=(.775,.74),color=LRT_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=0)
    axes[0,1].annotate("LREA-LS",xy=(.375,.575),color=LREA_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-40)



    for ax in axes[0,:]:
        ax.set_xticklabels([])

    for ax in axes[1,:]:
        ax.tick_params(axis="x",direction="out",pad=1)
        ax.set_xticklabels(["100","250","500","1000","1250","1500"],rotation=60)
        ax.set_xlabel(r"$|V_A|$")

    for ax in axes.reshape(-1):
        ax.set_ylim(0.0,1.00)
        ax.grid(True)

    axes[0,0].set_ylabel("accuracy")
    axes[1,0].set_ylabel("matched tris\n"+r"$\min{\{|T_A|,|T_B|\}}$",labelpad=0)
    xloc = -.26
    axes[1,0].annotate('', xy=(xloc, .2), xycoords='axes fraction', xytext=(xloc, 0.8),
                arrowprops=dict(arrowstyle="-", color='k'))

    #axes[0,0].set_yticklabels([])
    for ax in axes[:,1]:
        ax.set_yticklabels([])
    #axes[1,1].yaxis.set_label_position("right")


    for ax in axes[:,1]:
        ax.yaxis.tick_right()

    for ax in axes.reshape(-1):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)
        ax.set_ylim(-.00,1.00)
        ax.grid(True)


        ax.set_xlim(min(n_idx.keys()),max(n_idx.keys()))
        ax.set_xticks([100, 250, 500, 1000, 1250, 1500])
        ax.set_xlim(100,1500)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


#  Supplemental File

def RandomGeometricDupNoise_allModes(save_path=None):


    data_location = TAME_RESULTS + "RG_DupNoise/"
    
    fig= plt.figure(figsize=(5.75,4))
    n = 2
    m = 3
    spec = fig.add_gridspec(nrows=n, ncols=m,hspace=0.1,wspace=0.15,left=.125,right=.975,top=.975,bottom=.2)
    all_ax = []
    all_ax_gs = np.empty((n,m),object)

    def process_data(data,version="old"):

        p_idx = {p:i for (i,p) in enumerate(sorted(set([datum[1] for datum in data])))}
        n_idx = {n:i for (i,n) in enumerate(sorted(set([datum[2] for datum in data])))}
        sp_idx = {sp:i for (i,sp) in enumerate(sorted(set([datum[3] for datum in data])))}

        trials = int(len(data)/(len(p_idx)*len(n_idx)*len(sp_idx)))

        accuracy = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        klauAccuracy = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        triMatch = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        LT_klauTriMatch = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        trial_idx = np.zeros((len(p_idx),len(n_idx),len(sp_idx)),int)

        for datum in data:
            
            if version == "old":
                (seed,p,n,sp,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt) = datum
            elif version == "new klau":
                (seed,p,n,sp,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt,L_sparsity,fstatus) = datum
            else:
                print(f"only supports 'old' and 'new klau', got {version}")

            i = p_idx[p]
            j = n_idx[n]
            k = sp_idx[sp]
            accuracy[i,j,k,trial_idx[i,j,k]] = acc
            klauAccuracy[i,j,k,trial_idx[i,j,k]] = klau_acc
            triMatch[i,j,k,trial_idx[i,j,k]] = matched_tris/min(tri_A,tri_B)
            LT_klauTriMatch[i,j,k,trial_idx[i,j,k]] = klau_tri_match/min(tri_A,tri_B)
            trial_idx[i,j,k] += 1
        
        return accuracy, klauAccuracy, triMatch,LT_klauTriMatch, p_idx, n_idx, sp_idx  

    def make_percentile_plot(plot_ax, x_domain,data,color,hatch=None,**kwargs):   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
            (20,80,.05,color)
        ]
        
        n,m = data.shape
        percentile_linewidth=.01

        for (lower_percentile,upper_percentile,alpha,color) in ribbons:
  
            plot_ax.fill_between(x_domain,
                            np.percentile(data.T, lower_percentile, axis=0),
                            np.percentile(data.T, upper_percentile, axis=0),
                            facecolor=color,alpha=.1,edgecolor=color)

        for (col_func,alpha,color) in lines:
            line_data = []

            for i in range(n):
                line_data.append(col_func(data[i,:]))
        
            plot_ax.plot(x_domain,line_data,alpha=alpha,c=color,**kwargs)

   

    #hatches 
    LRT_Klau_hatch = None#"+"
    LT_Klau_hatch = None#"x"
    LRT_Tabu_hatch = None#"+"
    LT_Tabu_hatch = None#"+"
    
    

    default_p = .5
    default_n = 250
    default_sp = .25

    #
    #   p_edge experiments
    #
    sub_ax = [fig.add_subplot(spec[i,0]) for i in [0,1]]
    all_ax_gs[:,0] = sub_ax
    all_ax.append(sub_ax)
    
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        accuracy,  LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))

    LT_file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + LT_file,'r') as f:
        _,  LT_TabuAccuracy,_,LT_TabuTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))
    
    LRT_file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + LRT_file,'r') as f:
        LRT_accuracy,  LRT_TabuAccuracy,LRT_triMatch,LRT_TabuTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))
    
    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_KlauAccuracy,_,LRT_KlauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f),"new klau")


    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_lrm_KlauAccuracy,_,LRT_lrm_KlauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f),"new klau")


    file = "LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LREA_TabuAccuracy, _ ,LREA_TabuTriMatch, _,_,_ = process_data(json.load(f))

    file = "LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LREA_Accuracy,  LREA_KlauAccuracy, LREA_TriMatch ,LREA_KlauTriMatch, _,_,_ = process_data(json.load(f),"new klau")


    p_exps = [
        (accuracy,triMatch,LT_color,None,LT_linestyle),
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,LT_Klau_hatch,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,LT_Tabu_hatch,LT_Tabu_linestyle),
        
        (LRT_accuracy,LRT_triMatch,LRT_color,None,LRT_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,LRT_Tabu_hatch,LRT_Tabu_linestyle),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,LRT_Klau_hatch,LRT_Klau_linestyle),

        (LREA_Accuracy, LREA_TriMatch, LREigenAlign_color,None,LREigenAlign_linestyle),
        (LREA_TabuAccuracy, LREA_TabuTriMatch, LREA_Tabu_color,None,LREA_Tabu_linestyle),
        (LREA_KlauAccuracy, LREA_KlauTriMatch, LREA_Klau_color,None,LREA_Klau_linestyle),
        #(LRT_lrm_KlauAccuracy,LRT_lrm_KlauTriMatch,LREA_Klau_color,None,LREA_Klau_linestyle),
    ]

    for (acc, tri, c,hatch,linestyle) in p_exps:
        make_percentile_plot(sub_ax[0],p_idx.keys(),acc[:,n_idx[default_n],sp_idx[default_sp],:],c,hatch,linestyle=linestyle)
        make_percentile_plot(sub_ax[1],p_idx.keys(),tri[:,n_idx[default_n],sp_idx[default_sp],:],c,hatch,linestyle=linestyle)

    for ax in sub_ax:
        ax.set_xticks([0.0,.25,.5,.75,1.0])
        ax.set_xlim(min(p_idx.keys()),max(p_idx.keys()))

    sub_ax[1].set_xticklabels(["0.0",".25",r"${\bf .5}$",".75","1.0"],rotation=60)
    sub_ax[1].set_xlabel(r"$p_{edge}$")

 
    #
    #   n size experiments
    #
    sub_ax = [fig.add_subplot(spec[i,1]) for i in [0,1]]

    all_ax.append(sub_ax)
    all_ax_gs[:,1] = sub_ax

    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.75]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        accuracy,  LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))
    
    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json' 
    with open(data_location + file,'r') as f:
        _,  LT_TabuAccuracy,_,LT_TabuTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))
    
    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LRT_accuracy,  LRT_TabuAccuracy,LRT_triMatch,LRT_TabuTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))
    
    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_KlauAccuracy,_,LRT_KlauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f),"new klau")
    
    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_lrm_KlauAccuracy,_,LRT_lrm_KlauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f),"new klau")
    
    file ="LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LREA_Accuracy,  LREA_KlauAccuracy,LREA_TriMatch,LREA_KlauTriMatch, _, _, _ = process_data(json.load(f),"new klau")


    file ="LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LREA_TabuAccuracy,_,LREA_TabuTriMatch, _, _, _ = process_data(json.load(f))

    n_exps = [
        (accuracy,triMatch,LT_color,None,LT_linestyle),
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,LT_Klau_hatch,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,LT_Tabu_hatch,LT_Tabu_linestyle),
        
        (LRT_accuracy,LRT_triMatch,LRT_color,None,LRT_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,LRT_Tabu_hatch,LRT_Tabu_linestyle),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,LRT_Klau_hatch,LRT_Klau_linestyle),

        (LREA_Accuracy, LREA_TriMatch,LREigenAlign_color,None,LREigenAlign_linestyle),
        (LREA_KlauAccuracy, LREA_KlauTriMatch,LREA_Klau_color,None,LREA_Klau_linestyle),
        (LREA_TabuAccuracy,LREA_TabuTriMatch,LREA_Tabu_color,None,LREA_Tabu_linestyle),
        #(LRT_lrm_KlauAccuracy,LRT_lrm_KlauTriMatch,LRT_lrm_Klau_color,None,"solid"),
    ]

    for (acc, tri, c,hatch,linestyle) in n_exps:
        make_percentile_plot(sub_ax[0],n_idx.keys(),acc[p_idx[default_p],:,sp_idx[default_sp],:],c,hatch,linestyle=linestyle)
        make_percentile_plot(sub_ax[1],n_idx.keys(),tri[p_idx[default_p],:,sp_idx[default_sp],:],c,hatch,linestyle=linestyle)

    for ax in sub_ax:
        ax.set_xticks([100, 250, 500, 1000, 1250, 1500])
        ax.set_xlim(100,1500)

    
    sub_ax[1].set_xticklabels(["100",r"${\bf 250}$","500","1000","1250","1500"],rotation=60)
    sub_ax[1].set_xlabel(r"$|V_A|$")

 


    #
    #   step percentage experiments
    #

    sub_ax = [fig.add_subplot(spec[i,2]) for i in [0,1]]
    all_ax.append(sub_ax)
    all_ax_gs[:,2] = sub_ax
    
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        accuracy,  LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))


    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:TabuSearch_trialcount:20.json'
    with open(data_location + file,'r') as f:
        _,  LT_TabuAccuracy, _ ,LT_TabuTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LRT_accuracy,  LRT_TabuAccuracy, LRT_triMatch ,LRT_TabuTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_KlauAccuracy, _ ,LRT_KlauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f),"new klau")


    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
       #return json.load(f)
        _,  LRT_lrm_KlauAccuracy, _ ,LRT_lrm_KlauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f),version="new klau")

    file = "LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LREA_Accuracy,  LREA_TabuAccuracy, LREA_TriMatch ,LREA_TabuTriMatch,_,_,_ = process_data(json.load(f))
    
    file = "LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LREA_KlauAccuracy, _ ,LREA_KlauTriMatch,p_idx2, n_idx2, sp_idx2 = process_data(json.load(f),"new klau")

    sp_exps = [
        (accuracy,triMatch,LT_color,None,LT_linestyle),
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,LT_Klau_hatch,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,LT_Tabu_hatch,LT_Tabu_linestyle),

        (LRT_accuracy,LRT_triMatch,LRT_color,None,LRT_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,LRT_Tabu_hatch,LRT_Tabu_linestyle),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,LRT_Klau_hatch,LRT_Klau_linestyle),

        (LREA_Accuracy, LREA_TriMatch,LREigenAlign_color,None,LREigenAlign_linestyle),
        (LREA_TabuAccuracy, LREA_TabuTriMatch,LREA_Tabu_color,None,LREA_Tabu_linestyle),
        (LREA_KlauAccuracy, LREA_KlauTriMatch,LREA_Klau_color,None,LREA_Klau_linestyle),
        #(LRT_lrm_KlauAccuracy, LRT_lrm_KlauTriMatch,LRT_lrm_Klau_color,None,"solid")
    ]
    

    for (acc, tri, c,hatch,linestyle) in sp_exps:
        make_percentile_plot(sub_ax[0],sp_idx.keys(),acc[p_idx[default_p],n_idx[default_n],:,:],c,hatch,linestyle=linestyle)
        make_percentile_plot(sub_ax[1],sp_idx.keys(),tri[p_idx[default_p],n_idx[default_n],:,:],c,hatch,linestyle=linestyle)
  
    for ax in sub_ax:
        ax.set_xticks([.05,.1,.25,.5])
        ax.set_xlim(0.05,.5)
        
    sub_ax[1].set_xticklabels(["5%","10%",r"${\bf 25\%}$","50%"],rotation=60)#52.5
    
    shift = -.25
    sub_ax[1].annotate(r"$|V_B|-|V_A|$",xy=(.5,-.09+shift),xycoords='axes fraction',ha="center")
    sub_ax[1].annotate('', xy=(.23, -.125+shift), xycoords='axes fraction', xytext=(.77, -.125+shift),
                       arrowprops=dict(arrowstyle="-", color='k',linewidth=.5))
    sub_ax[1].annotate(r"$|V_A|$",xy=(.5,-.21+shift),xycoords='axes fraction',ha="center")
    sub_ax[1].annotate(r"(%)",xy=(.78, -.125+shift),xycoords='axes fraction',ha="left",va="center")
   

    #
    #  Final Axes touch up 
    #

    all_ax_gs[0,0].set_ylabel("accuracy")
    all_ax_gs[1,0].set_ylabel("matched tris\n"+r"$\min{\{|T_A|,|T_B|\}}$",labelpad=1)
    xloc = -.31
    all_ax_gs[1,0].annotate('', xy=(xloc , .18), xycoords='axes fraction', xytext=(xloc, 0.82),
                arrowprops=dict(arrowstyle="-", color='k'))
    for ax in all_ax_gs[0,:]:
        ax.set_xticklabels([])

    for ax in all_ax_gs[1,:]:
        ax.tick_params(axis="x",direction="out",pad=1)
        
    for ax in all_ax_gs[:,1:].reshape(-1):
        ax.set_yticklabels([])

    #all_ax_gs[0,0].set_yticklabels([])
    #all_ax_gs[1,0].set_yticklabels([])


    #for ax in all_ax_gs[:,2]:
    #    ax.yaxis.tick_right()

    for ax in all_ax_gs.reshape(-1):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)
        ax.set_ylim(0.0,1.0)
        ax.grid(True)

    all_ax_gs[1,1].annotate(r"$\Lambda$-TAME",xy=(.4,.26),color=LT_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-15)
    all_ax_gs[0,2].annotate(r"$\Lambda$T" +"-Klau",xy=(.65,.875),color=LT_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-17.5)
    all_ax_gs[1,2].annotate(r"$\Lambda$T" +"-LS",xy=(.825,.9),color=LT_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-7)

    all_ax_gs[1,2].annotate("LRT-TAME",xy=(.75,.15),color=LRT_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-7.5)
    all_ax_gs[1,1].annotate(r"LRT-LS",xy=(.8,.85),color=LRT_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10)
    all_ax_gs[1,2].annotate(r"LRT-Klau",xy=(.3,.675),color=LRT_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-35)

    all_ax_gs[1,0].annotate("LREA",xy=(.325,.075),color=LREigenAlign_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-10)
    all_ax_gs[1,1].annotate(r"LREA-LS",xy=(.5,.65),color=LREA_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-25)
    all_ax_gs[1,1].annotate(r"LREA-Klau",xy=(.7,.3),color=LREA_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-30)


    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def RandomGeometricERNoise_allModes(save_path=None):

    data_location = TAME_RESULTS + "RG_ERNoise/"
    fig = plt.figure(figsize=(4.25,4.25))

    n = 2
    m = 2

    spec = fig.add_gridspec(nrows=n, ncols=m,hspace=0.1,wspace=0.15,left=.15,right=.97,top=.975,bottom=.2)
    all_ax = []
    axes = np.empty((n,m),object)
    for i in range(n):
        for j in range(m):
            ax = fig.add_subplot(spec[i,j])
            axes[i,j] = ax


    def process_data(data,version="klau_old"):
        p_idx = {p:i for (i,p) in enumerate(sorted(set([datum[1] for datum in data])))}
        n_idx = {p:i for (i,p) in enumerate(sorted(set([datum[2] for datum in data])))}
        trials = int(len(data)/(len(p_idx)*len(n_idx)))

        accuracy = np.zeros((len(p_idx),len(n_idx),trials))
        LT_klauAccuracy = np.zeros((len(p_idx),len(n_idx),trials))
        triMatch = np.zeros((len(p_idx),len(n_idx),trials))
        LT_klauTriMatch = np.zeros((len(p_idx),len(n_idx),trials))
        trial_idx = np.zeros((len(p_idx),len(n_idx)),int)


        for datum in data:

            if version == "klau_new":
                (seed,p,n,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt,L_sparsity,fbounds) = datum
            elif version == "klau_old":
                (seed,p,n,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt) = datum
            elif version == "tabu":
                (seed,p,n,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt) = datum
            else:
                raise ValueError("only supports ")
            i = p_idx[p]
            j = n_idx[n]

            accuracy[i,j,trial_idx[i,j]] = acc
            LT_klauAccuracy[i,j,trial_idx[i,j]] = klau_acc
            triMatch[i,j,trial_idx[i,j]] = matched_tris/min(tri_A,tri_B)
            LT_klauTriMatch[i,j,trial_idx[i,j]] = klau_tri_match/min(tri_A,tri_B)
            trial_idx[i,j] += 1
        
        return accuracy, LT_klauAccuracy, triMatch, LT_klauTriMatch, p_idx, n_idx 

    def make_percentile_plot(plot_ax, x_domain,data,color,hatch=None,**kwargs):
        #TODO: check on scope priorities of ax for closures   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
            (20,80,.05,color)
        ]
        
        #plot_percentiles(plot_ax,  data.T, x_domain, lines, ribbons,**kwargs)
        n,m = data.shape
        percentile_linewidth=.01

        for (lower_percentile,upper_percentile,alpha,color) in ribbons:
            #plot_ax.plot(np.percentile(data.T, lower_percentile, axis=0),c=color,linewidth=percentile_linewidth)
            #plot_ax.plot(np.percentile(data.T, upper_percentile, axis=0),c=color,linewidth=percentile_linewidth)

            plot_ax.fill_between(x_domain,
                            np.percentile(data.T, lower_percentile, axis=0),
                            np.percentile(data.T, upper_percentile, axis=0),
                            facecolor=color,alpha=.1,edgecolor=color)
            """
            plot_ax.fill_between(x_domain,
                            np.percentile(data.T, lower_percentile, axis=0),
                            np.percentile(data.T, upper_percentile, axis=0),
                            facecolor="None",hatch=hatch,edgecolor=color,alpha=.4)
            """

        for (col_func,alpha,color) in lines:
            line_data = []

            for i in range(n):
                line_data.append(col_func(data[i,:]))
        
            plot_ax.plot(x_domain,line_data,alpha=alpha,c=color,**kwargs)



    #
    #   p_remove experiments
    #
    sub_ax = axes[:,0]

    #file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.0,0.005,0.01,0.05,0.1,0.2]_postProcess:KlauAlgo_trialcount:20.json"
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        accuracy, LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx = process_data(json.load(f),"klau_new")

    files=[
    #    "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.0,0.005,0.01,0.05,0.1,0.2]_postProcess:TabuSearch_trialcount:20.json",
        "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.05,0.15,0.25,0.3,0.4,0.5]_postProcess:TabuSearch_trialcount:20.json",
        "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.01,0.1,0.2]_postProcess:TabuSearch_trialcount:20.json"
    ]

    res = []
    for file in files:
        with open(data_location + file,'r') as f:
            res.extend(json.load(f))
    
    _,  LT_TabuAccuracy, _, LT_TabuTriMatch, p_idx, n_idx = process_data(res,"tabu")


    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        LRT_accuracy,  LRT_KlauAccuracy, LRT_triMatch, LRT_KlauTriMatch, p_idx, n_idx = process_data(json.load(f),"klau_new")

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        _, LRT_TabuAccuracy,_,LRT_TabuTriMatch, p_idx, n_idx = process_data(json.load(f),"tabu")


    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        _, LRT_lrm_klauAccuracy,_,LRT_lrm_klauTriMatch, p_idx, n_idx = process_data(json.load(f),"klau_new")

    file = "LowRankEigenAlign_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        LREA_accuracy, LREA_klauAccuracy,LREA_triMatch,LREA_klauTriMatch, _ ,_ = process_data(json.load(f),"klau_new")

    file = "LowRankEigenAlign_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        _, LREA_tabuAccuracy,_,LREA_tabuTriMatch, _,_ = process_data(json.load(f),"tabu")



    #make_percentile_plot(sub_ax[0],p_idx.keys(),LT_TabuAccuracy[:,0,:],LT_Tabu_color)
    #make_percentile_plot(sub_ax[1],p_idx.keys(),LT_TabuTriMatch[:,0,:],LT_Tabu_color)

    p_exps = [
        (accuracy,triMatch,LT_color,LT_linestyle),
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,LT_Tabu_linestyle),
        
        #(LRT_lrm_klauAccuracy,LRT_lrm_klauTriMatch,LRM_lrm_Klau_color),
        (LRT_accuracy,LRT_triMatch,LRT_color,LRT_linestyle),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,LRT_Klau_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,LRT_Tabu_linestyle),

        (LREA_accuracy,LREA_triMatch,LREigenAlign_color,LREigenAlign_linestyle),
        (LREA_tabuAccuracy,LREA_tabuTriMatch,LREA_Tabu_color,LREA_Tabu_linestyle),
        (LREA_klauAccuracy,LREA_klauTriMatch,LREA_Klau_color,LREA_Klau_linestyle),
    ]

    for (acc,tri,c,linestyle) in p_exps:
        make_percentile_plot(sub_ax[0],p_idx.keys(),acc[:,0,:],c,linestyle=linestyle)
        make_percentile_plot(sub_ax[1],p_idx.keys(),tri[:,0,:],c,linestyle=linestyle)

  

    for ax in sub_ax:
        ax.set_xticks([0.01, 0.05, 0.1, 0.2,.3,.4])
        ax.set_xlim(min(p_idx.keys()),.4)
        
    
    sub_ax[1].set_xticklabels([1.e-02, r"${\bf 0.05}$", .1, .2,.3,.4],rotation=60)

    sub_ax[1].set_xlabel(r"$p_{remove} \equiv  p$"+'\n'+r"$p_{add}=\frac{p\rho}{1-\rho}$",ha="center",labelpad=-5)

    #
    #   n size experiments
    #

    sub_ax = axes[:,1]
    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json'
    with open(data_location + file,'r') as f:
        accuracy,  LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx = process_data(json.load(f),"klau_new")

    #file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.01]_postProcess:TabuSearch_trialcount:20.json'
    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:TabuSearch_trialcount:20.json'
    with open(data_location + file,'r') as f:
        #return json.load(f)
        _,  LT_TabuAccuracy, _, LT_TabuTriMatch, p_idx, n_idx = process_data(json.load(f),"tabu")

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        LRT_accuracy,  LRT_KlauAccuracy, LRT_triMatch, LRT_KlauTriMatch, p_idx, n_idx = process_data(json.load(f),"klau_new")

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        _, LRT_TabuAccuracy,_,LRT_TabuTriMatch, p_idx, n_idx = process_data(json.load(f),"tabu")


    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        _,  LRT_lrm_KlauAccuracy, _, LRT_lrm_KlauTriMatch, p_idx, n_idx = process_data(json.load(f),"klau_new")

    

    file = "LowRankEigenAlign_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        LREA_accuracy, LREA_klauAccuracy,LREA_triMatch,LREA_klauTriMatch, _ ,_ = process_data(json.load(f),"klau_new")

    file = "LowRankEigenAlign_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        _, LREA_tabuAccuracy,_,LREA_tabuTriMatch, _,_ = process_data(json.load(f),"tabu")


    n_exps = [
        (accuracy,triMatch,LT_color,LT_linestyle),
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,LT_Tabu_linestyle),
        #(LRT_lrm_KlauAccuracy,LRT_lrm_KlauTriMatch,LRM_lrm_Klau_color),

        (LRT_accuracy,LRT_triMatch,LRT_color,LRT_linestyle),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,LRT_Klau_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,LRT_Tabu_linestyle),

        (LREA_accuracy,LREA_triMatch,LREigenAlign_color,LREigenAlign_linestyle),
        (LREA_tabuAccuracy,LREA_tabuTriMatch,LREA_Tabu_color,LREA_Tabu_linestyle),
        (LREA_klauAccuracy,LREA_klauTriMatch,LREA_Klau_color,LREA_Klau_linestyle),
    ]

    for (acc,tri,c,linestyle) in n_exps:
        make_percentile_plot(sub_ax[0],n_idx.keys(),acc[0,:,:],c,linestyle=linestyle)
        make_percentile_plot(sub_ax[1],n_idx.keys(),tri[0,:,:],c,linestyle=linestyle)


    for ax in sub_ax:
        ax.set_xlim(min(n_idx.keys()),max(n_idx.keys()))
        ax.set_xticks([100, 250, 500, 1000, 1250, 1500])
  

    sub_ax[1].set_xticklabels(["100",r"${\bf 250}$","500","1000","1250","1500"],rotation=60)
    sub_ax[1].set_xlabel(r"$|V_A|$")
    
    axes[1,1].annotate(r"$\Lambda$-TAME",xy=(.175,.35),color=LT_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-40)
    axes[1,0].annotate(r"$\Lambda$T-"+"Klau",xy=(.45,.35),color=LT_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-45)
    axes[0,0].annotate(r"$\Lambda$T-"+"LS",xy=(.475,.425),color=LT_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-50)
    
    axes[0,1].annotate("LR-TAME",xy=(.825,.17),color=LRT_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-2.5)
    axes[0,1].annotate("LRT-Klau",xy=(.325,.475),color=LRT_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=0)
    axes[0,1].annotate("LRT-LS",xy=(.85,.73),color=LRT_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-10)

    axes[1,1].annotate("LREA",xy=(.125,.075),color=LREigenAlign_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-20)
    axes[1,1].annotate("LREA-Klau",xy=(.6,.525),color=LREA_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-25)
    axes[1,0].annotate("LREA-LS",xy=(.8,.4),color=LREA_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-10)


    axes[0,0].set_ylabel("accuracy")
    axes[1,0].set_ylabel("matched tris\n"+r"$\min{\{|T_A|,|T_B|\}}$",labelpad=2.5)
    xloc = -.3
    axes[1,0].annotate('', xy=(xloc, .2), xycoords='axes fraction', xytext=(xloc, 0.8),
                arrowprops=dict(arrowstyle="-", color='k'))
    
    for ax in axes[0,:]:
        ax.set_xticklabels([])
    for ax in axes[:,1]:
        ax.set_yticklabels([])

    for ax in axes[1,:]:
        ax.tick_params(axis="x",direction="out",pad=1)

    for ax in axes[:,1]:
        ax.yaxis.tick_right()

    for ax in axes.reshape(-1):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)
        ax.set_ylim(0.0,1.0)
        ax.grid(True)


    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)



#
#   K nearest Neighbors Post Processing Experiments
#
def LambdaTAME_increasing_clique_size(save_path=None):

    plt.rc('text.latex', preamble=r'\usepackage{/Users/charlie/Documents/Code/TKPExperimentPlots/latex/dgleich-math}')
    
    fig = plt.figure(figsize=(6,3.5))
    global_ax = plt.gca()


    data_location = TAME_RESULTS + "klauExps/"

    data_location = TAME_RESULTS + "RG_DupNoise/"
    def process_klau_data(data,version="LT"):
        order_idx = {order:i for (i,order) in enumerate(sorted(set([datum[0] for datum in data])))}
        k_idx = {k:i for (i,k) in enumerate(sorted(set([datum[1] for datum in data])))}
        trials =len(data[0][-1])

        runtime = np.zeros((len(order_idx),len(k_idx),trials))
        accuracy = np.zeros((len(order_idx),len(k_idx),trials))
        postProcessingAccuracy = np.zeros((len(order_idx),len(k_idx),trials))
        triMatch = np.zeros((len(order_idx),len(k_idx),trials))
        postProcessingRuntime = np.zeros((len(order_idx),len(k_idx),trials))
        #sparsity = np.zeros((len(order_idx),len(k_idx),trials))
        trial_idx = np.zeros((len(order_idx),len(k_idx)),int)

        vertex_coverage = {}
        A_motifs_counts = {}


        for (order,k,results) in data:
            # expecting results:
            #     (seed,p, n, sp, acc, dup_tol_acc, matched_matching_score, A_motifs, B_motifs, A_motifDistribution, B_motifsDistribution,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt)
            i = order_idx[order]
            j = k_idx[k]

    
            accuracy[i,j,:] = [x[4] for x in results]
            if version=="LT":
                runtime[i,j,:] = [sum([sum(val) for val in result[11].values()]) for result in results] 
                postProcessingAccuracy[i,j,:] = [x[15] for x in results]
                postProcessingRuntime[i,j,:] = [x[17] + x[18] for x in results]
                     #sparsity[i,j,:] = [x[19] for x in results]
                if k == min(k_idx.keys()): #only do this once over k 

                    #determine the size of A vs. B motif data
                    if len(results[0][9]) == results[0][2]:
                        vertex_coverage[order] = [len(list(filter(lambda y: y!= 0.0,x[9][0])))/x[2] for x in results]
                        A_motifs_counts[order]= [x[7][0] for x in results]
                    else:
                        vertex_coverage[order] = [len(list(filter(lambda y: y!= 0.0,x[10][0])))/x[2] for x in results]
                        A_motifs_counts[order]= [x[8][0] for x in results]


            elif version=="LRT":
                postProcessingAccuracy[i,j,:] = [x[14] for x in results]
                postProcessingRuntime[i,j,:] = [x[16] + x[17] for x in results]

        if version=="LT":
            return accuracy,runtime, postProcessingAccuracy, postProcessingRuntime,vertex_coverage,A_motifs_counts, order_idx, k_idx #triMatch,LT_klauTriMatch, p_idx, n_idx 
        else:
            return accuracy, postProcessingAccuracy, postProcessingRuntime,order_idx, k_idx #triMatch,LT_klauTriMatch, p_idx, n_idx 
    
    def process_tabu_data(data,version="LT"):
        order_idx = {order:i for (i,order) in enumerate(sorted(set([datum[0] for datum in data])))}
        k_idx = {k:i for (i,k) in enumerate(sorted(set([datum[1] for datum in data])))}
        trials =len(data[0][-1])

        accuracy = np.zeros((len(order_idx),len(k_idx),trials))
        tabuAccuracy = np.zeros((len(order_idx),len(k_idx),trials))
        triMatch = np.zeros((len(order_idx),len(k_idx),trials))
        tabuRuntime = np.zeros((len(order_idx),len(k_idx),trials))
        #sparsity = np.zeros((len(order_idx),len(k_idx),trials))
        trial_idx = np.zeros((len(order_idx),len(k_idx)),int)

        vertex_coverage = {}



        for (order,k,results) in data:
            # expecting results:
            #     (seed,p, n, sp, acc, dup_tol_acc, matched_matching_score, A_motifs, B_motifs, A_motifDistribution, B_motifsDistribution,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt)
            i = order_idx[order]
            j = k_idx[k]
            accuracy[i,j,:] = [x[4] for x in results]
            if version == "LT":
                tabuAccuracy[i,j,:] = [x[15] for x in results]
                tabuRuntime[i,j,:] = [x[18] for x in results]
                #sparsity[i,j,:] = [x[19] for x in results]
                if k == min(k_idx.keys()): #only do this once over k 
                    vertex_coverage[order] = [len(list(filter(lambda y: y!= 0.0,x[9][0])))/x[2] for x in results]
                    

            elif version == "LRT":
                print(i," ",j)
                tabuAccuracy[i,j,:] = [x[14] for x in results]
                tabuRuntime[i,j,:] = [x[17] for x in results]

        if version == "LT":
            return accuracy, tabuAccuracy,tabuRuntime,vertex_coverage, order_idx, k_idx
        else:
            return accuracy, tabuAccuracy,tabuRuntime,order_idx, k_idx
    

    filename = "RandomGeometric_degreedist:LogNormal_KlauAlgokvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_KAmiter:1000_orders:[2,3,4,5,6,7,8,9]_p:[0.5]_samples:1000000_sp:[0.25]_trials:25_data.json"
    with open(data_location+filename,'r') as f:
        accuracy,LT_runtime, LT_klauAccuracy,LT_klauRuntime,vertex_coverage,A_motifs_counts, order_idx, k_idx = process_klau_data(json.load(f))

    filename = "RandomGeometric_degreedist:LogNormal_TabuSearchkvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_p:[0.5]_samples:1000000_sp:[0.25]_trials:25_data.json"
    with open(data_location+filename,'r') as f:
        _, LT_TabuAccuracy, LT_TabuRuntime, _, order_idx, k_idx = process_tabu_data(json.load(f))

    def make_percentile_plot(plot_ax, x_domain,data,color,**kwargs):   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
            (20,80,.2,color)
        ]
        plot_percentiles(plot_ax,  data.T, x_domain, lines, ribbons,**kwargs)

    def make_violin_plot(ax,data,precision=2,c=None,v_alpha=.8):

        background_v = ax.violinplot(data, points=100, positions=[0.5], showmeans=False, 
                        showextrema=False, showmedians=False,widths=.5,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        v = ax.violinplot(data, points=100, positions=[.5], showmeans=False, 
                       showextrema=False, showmedians=True,widths=.5,vert=False)#,quantiles=[[.2,.8]]*len(data2))

        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        newMedianLines = [[(x0,y0-.1),(x0,y0 + (y1-y0)/2 -.1)],[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
        v["cmedians"].set_segments(newMedianLines)

        ax.annotate(f"{np.median(data):.{precision}f}",xy=(.5,.38),xycoords="axes fraction",ha="center",fontsize=10)
        ax.annotate(f"{np.min(data):.{precision}f}",xy=(0,.8),xycoords="axes fraction",ha="left",fontsize=6,alpha=.8)
        ax.annotate(f"{np.max(data):.{precision}f}",xy=(1,.1),xycoords="axes fraction",ha="right",fontsize=6,alpha=.8)

        if c is not None:
            
            v["cmedians"].set_color("k")
            v["cmedians"].set_alpha(.3)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor(c)            
                b.set_alpha(v_alpha)
                #b.set_alpha(1)
                b.set_color(c)
            
            for b in background_v["bodies"]:
                b.set_facecolor("w")
                b.set_edgecolor("w")
                b.set_color("w")



    def make_violin_plot_legend(ax,c="k"):
        
        np.random.seed(12)
        v = ax.violinplot([np.random.normal() for i in range(50)], points=100, positions=[.5], showmeans=False, 
                        showextrema=False, showmedians=True,widths=.5,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #ax.axes.set_axis_off()

        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        newMedianLines = [[(x0,y0-.1),(x0,y0 + (y1-y0)/2 -.1)],[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
        v["cmedians"].set_segments(newMedianLines)

        ax.annotate("med.",xy=(.5,.5),xycoords="axes fraction",ha="center",va="center",fontsize=10)
        ax.annotate(f"min",xy=(0,.8),xycoords="axes fraction",ha="left",fontsize=6,alpha=.8)
        ax.annotate(f"max",xy=(1,.1),xycoords="axes fraction",ha="right",fontsize=6,alpha=.8)

        if c is not None:
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor(c)
                b.set_alpha(.2)
                b.set_color(c)

    #  -- add in order labels overhead --  #

    global_ax.set_yticklabels([])
    global_ax.set_yticklabels([])


    widths = [.5, 3, 2, .8,.8]
    spec = fig.add_gridspec(nrows=5,ncols=1+len(order_idx),hspace=0.0,wspace=0.0,height_ratios=widths,left=.15,right=.95)

    allCAx = []
    allAccAx = [] 
    allRtAx = [] 
    allVCAx = []
    allMotifCountAx = []
    allSparsityAx = []
    first = True
    annotation_idx = 4

    if filename == "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json":
        k_tick_idx = 0
    else:
        k_tick_idx = 4
    parity = 0
    for idx,(order,i) in enumerate(order_idx.items()):
        #
        #  Clique size
        #
        ax = fig.add_subplot(spec[0,i])
        
        allCAx.append(ax)
        if idx % 2 == parity:
            ax.patch.set_facecolor('k')
            ax.patch.set_alpha(0.1)
        ax.annotate(f"{order}",xy=(.5, .5), xycoords='axes fraction', c="k",size=10,ha="center",va="center")

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        #
        #  Accuracy Plots
        #
        if idx == annotation_idx:
            ax = fig.add_subplot(spec[1,i],zorder=5)#,sharey=allAccAx[0])
        else:
            ax = fig.add_subplot(spec[1,i])#,sharey=allAccAx[0])
        allAccAx.append(ax)

        if idx % 2 != parity:
            ax.patch.set_facecolor('k')
            ax.patch.set_alpha(0.1)
        
        ax.set_yticks([0.0,0.25,.5,.75,1.0])
        ax.set_ylim(0,1.0)

                
        plt.axhline(y=np.median(accuracy[i,:,:]), color=LT_color,linestyle=LT_linestyle)
        plt.axhline(y=np.max(accuracy[i,:,:]), color=LT_color,linestyle="dotted")
        make_percentile_plot(ax,k_idx.keys(),LT_klauAccuracy[i,:,:],LT_Klau_color)#,linestyle=LT_Klau_linestyle
        make_percentile_plot(ax,k_idx.keys(),LT_TabuAccuracy[i,:,:],LT_Tabu_color)#,linestyle=LT_Tabu_linestyle

        ax.set_xticklabels([])
        ax.set_yticklabels([])


        #
        #  Runtime Plots
        #
        
        if idx == k_tick_idx:
            ax = fig.add_subplot(spec[2,i],zorder=3)
        else:
            ax = fig.add_subplot(spec[2,i])
        allRtAx.append(ax)
        
        if idx % 2 == parity:
            ax.patch.set_facecolor('k')
            ax.patch.set_alpha(0.1)

        if filename == "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json":
            ax.set_ylim(np.min(LT_klauRuntime[:,:,:]),600)
        elif filename == "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json":
            ax.set_ylim(np.min(LT_klauRuntime[:,:,:]),1200)
        else:
            ax.set_ylim(np.min(LT_klauRuntime[:,:,:]),2100)
        #ax.set_yscale("log")
        if idx != 0:
            ax.set_yticklabels([])
        ax.set_xticks([15,45,90])

        ax.set_xticklabels([])
            
        make_percentile_plot(ax,k_idx.keys(),LT_klauRuntime[i,:,:],LT_Klau_color,linestyle=LT_Klau_linestyle)
        make_percentile_plot(ax,k_idx.keys(),LT_TabuRuntime[i,:,:],LT_Tabu_color,linestyle=LT_Tabu_linestyle)

        plt.axhline(y=np.median(LT_runtime[i,:,:]), color=LT_color,linestyle=LT_linestyle)


        #
        #  Vertex Coverage
        #
        ax = fig.add_subplot(spec[3,i])
        
        allVCAx.append(ax)

        
        if idx % 2 != parity:
            make_violin_plot(ax,vertex_coverage[order],c="w")
            ax.patch.set_facecolor('k')
            ax.patch.set_alpha(0.1)
        else:
            make_violin_plot(ax,vertex_coverage[order],c="k",v_alpha=.1)


        #
        #  Motif Counts
        #
        
        ax = fig.add_subplot(spec[4,i])
        
        allMotifCountAx.append(ax)

        if idx % 2 == parity:
            make_violin_plot(ax,A_motifs_counts[order],c="w",precision=0)
            ax.patch.set_facecolor('k')
            ax.patch.set_alpha(0.1)
        else:
            make_violin_plot(ax,A_motifs_counts[order],c="k",v_alpha=.1,precision=0)



    violinLegendAx = fig.add_subplot(spec[3,-1])

    for ax in chain(allAccAx,allRtAx,allCAx,allVCAx,allMotifCountAx,allSparsityAx,[global_ax],[violinLegendAx]):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for ax in chain(allAccAx,allRtAx,allSparsityAx):
        ax.grid(True)
        ax.set_xticks([15,45,90])
        ax.set_xlim(min(k_idx.keys()),max(k_idx.keys()))

    #  - make violin plot legend -  #
    make_violin_plot_legend(violinLegendAx)

    #  -- Add in x-domain annotations mid plot --  #

    allRtAx[k_tick_idx].xaxis.set_label_position("top")
    allRtAx[k_tick_idx].xaxis.set_ticks_position('top')
    allRtAx[k_tick_idx].tick_params(axis="x",direction="out", pad=-15)
    allAccAx[k_tick_idx].annotate("nearest neighbors ("+r"$K$"+')',xy=(.5,.05),ha="center",xycoords='axes fraction')

    allRtAx[k_tick_idx].set_xticklabels([15,45,90],zorder=5)




    #  -- Alternate tick labels to opposite axes --  #


    allCAx[0].annotate("Clique Size",xy=(-.1,.5),ha="right",va="center",xycoords='axes fraction')

    allAccAx[-1].yaxis.set_ticks_position('right')
    allAccAx[-1].set_yticklabels([0,.25,.5,.75,1.0])
    allAccAx[-1].tick_params(axis="both",direction="out",which='both', length=7.5)
    allAccAx[0].set_ylabel("Accuracy",rotation=0,labelpad=0,ha="right")
    
    allRtAx[-1].yaxis.set_label_position("right")

    allRtAx[0].annotate("Runtime (s)",xy=(-.1,.5),ha="right",va="center",xycoords='axes fraction')
    allRtAx[-1].yaxis.set_ticks_position('right')
    if filename == "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json":
        allRtAx[0].set_yticklabels(["10 s","2 min","5 min","10 min"])
        for ax in allRtAx:
            ax.set_yticks([1e0,1e1,120,300,600])   
    elif filename == "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json":
        allRtAx[0].set_yticklabels(["10 s","5 min","10 min","15 min","20 min"])
        for ax in allRtAx:
            ax.set_yticks([1e1,300,600,900,1200])
    else:
        for i,ax in enumerate(allRtAx):
            
            ax.set_yscale("log")

            ax.set_yticks([1e0,1e1,1e2,1e3,1e4])
            ax.set_ylim(1e-1,2e4)
            if i == len(allRtAx)-1:
                ax.set_yticklabels([r"$10^0$",r"$10^1$",r"$10^2$",r"$10^3$",None])
            else:
                ax.set_yticklabels([])


    allVCAx[0].annotate("Vertex\nCoverage",xy=(-.1,.5),ha="right",va="center",xycoords='axes fraction')
    #  -- Annotate accuracy plots --  #
    allAccAx[5].annotate(r"$\Lambda$T"+"-Klau", xy=(.5, .675), xycoords='axes fraction', c=LT_Klau_color,size=10,ha="center",rotation=20)
    allAccAx[5].annotate(r"$\Lambda$T"+"-LS", xy=(.575, .5), xycoords='axes fraction', c=LT_Tabu_color,size=10,ha="center",zorder=4)
    allAccAx[annotation_idx].annotate("maximum "+r"$\Lambda$"+"T", xy=(.975, .45), xycoords='axes fraction', c=LT_color,size=10,ha="right")
    allAccAx[annotation_idx].annotate("median "+r"$\Lambda$"+"T", xy=(.975, .275), xycoords='axes fraction', c=LT_color,size=10,ha="right")

    allMotifCountAx[0].annotate("A Motifs",xy=(-.1,.5),ha="right",va="center",xycoords='axes fraction')
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def LambdaTAME_increasing_clique_size_v2(save_path=None):

    plt.rc('text.latex', preamble=r'\usepackage{/Users/charlie/Documents/Code/TKPExperimentPlots/latex/dgleich-math}')
    
    fig = plt.figure(figsize=(6,3.5))
    global_ax = plt.gca()


    data_location = TAME_RESULTS + "klauExps/"

    data_location = TAME_RESULTS + "RG_DupNoise/"
    def process_klau_data(data,version="LT"):
        order_idx = {order:i for (i,order) in enumerate(sorted(set([datum[0] for datum in data])))}
        k_idx = {k:i for (i,k) in enumerate(sorted(set([datum[1] for datum in data])))}
        trials =len(data[0][-1])

        runtime = np.zeros((len(order_idx),len(k_idx),trials))
        accuracy = np.zeros((len(order_idx),len(k_idx),trials))
        postProcessingAccuracy = np.zeros((len(order_idx),len(k_idx),trials))
        triMatch = np.zeros((len(order_idx),len(k_idx),trials))
        postProcessingRuntime = np.zeros((len(order_idx),len(k_idx),trials))
        #sparsity = np.zeros((len(order_idx),len(k_idx),trials))
        trial_idx = np.zeros((len(order_idx),len(k_idx)),int)

        vertex_coverage = {}
        A_motifs_counts = {}


        for (order,k,results) in data:
            # expecting results:
            #     (seed,p, n, sp, acc, dup_tol_acc, matched_matching_score, A_motifs, B_motifs, A_motifDistribution, B_motifsDistribution,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt)
            i = order_idx[order]
            j = k_idx[k]

    
            accuracy[i,j,:] = [x[4] for x in results]
            if version=="LT":
                runtime[i,j,:] = [sum([sum(val) for val in result[11].values()]) for result in results] 
                postProcessingAccuracy[i,j,:] = [x[15] for x in results]
                postProcessingRuntime[i,j,:] = [x[17] + x[18] for x in results]
                     #sparsity[i,j,:] = [x[19] for x in results]
                if k == min(k_idx.keys()): #only do this once over k 

                    #determine the size of A vs. B motif data
                    if len(results[0][9]) == results[0][2]:
                        vertex_coverage[order] = [len(list(filter(lambda y: y!= 0.0,x[9][0])))/x[2] for x in results]
                        A_motifs_counts[order]= [x[7][0] for x in results]
                    else:
                        vertex_coverage[order] = [len(list(filter(lambda y: y!= 0.0,x[10][0])))/x[2] for x in results]
                        A_motifs_counts[order]= [x[8][0] for x in results]


            elif version=="LRT":
                postProcessingAccuracy[i,j,:] = [x[14] for x in results]
                postProcessingRuntime[i,j,:] = [x[16] + x[17] for x in results]

        if version=="LT":
            return accuracy,runtime, postProcessingAccuracy, postProcessingRuntime,vertex_coverage,A_motifs_counts, order_idx, k_idx #triMatch,LT_klauTriMatch, p_idx, n_idx 
        else:
            return accuracy, postProcessingAccuracy, postProcessingRuntime,order_idx, k_idx #triMatch,LT_klauTriMatch, p_idx, n_idx 
    
    def process_tabu_data(data,version="LT"):
        order_idx = {order:i for (i,order) in enumerate(sorted(set([datum[0] for datum in data])))}
        k_idx = {k:i for (i,k) in enumerate(sorted(set([datum[1] for datum in data])))}
        trials =len(data[0][-1])

        accuracy = np.zeros((len(order_idx),len(k_idx),trials))
        tabuAccuracy = np.zeros((len(order_idx),len(k_idx),trials))
        triMatch = np.zeros((len(order_idx),len(k_idx),trials))
        tabuRuntime = np.zeros((len(order_idx),len(k_idx),trials))
        #sparsity = np.zeros((len(order_idx),len(k_idx),trials))
        trial_idx = np.zeros((len(order_idx),len(k_idx)),int)

        vertex_coverage = {}



        for (order,k,results) in data:
            # expecting results:
            #     (seed,p, n, sp, acc, dup_tol_acc, matched_matching_score, A_motifs, B_motifs, A_motifDistribution, B_motifsDistribution,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt)
            i = order_idx[order]
            j = k_idx[k]
            accuracy[i,j,:] = [x[4] for x in results]
            if version == "LT":
                tabuAccuracy[i,j,:] = [x[15] for x in results]
                tabuRuntime[i,j,:] = [x[18] for x in results]
                #sparsity[i,j,:] = [x[19] for x in results]
                if k == min(k_idx.keys()): #only do this once over k 
                    vertex_coverage[order] = [len(list(filter(lambda y: y!= 0.0,x[9][0])))/x[2] for x in results]
                    

            elif version == "LRT":
                print(i," ",j)
                tabuAccuracy[i,j,:] = [x[14] for x in results]
                tabuRuntime[i,j,:] = [x[17] for x in results]

        if version == "LT":
            return accuracy, tabuAccuracy,tabuRuntime,vertex_coverage, order_idx, k_idx
        else:
            return accuracy, tabuAccuracy,tabuRuntime,order_idx, k_idx
    

    filename = "RandomGeometric_degreedist:LogNormal_KlauAlgokvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_KAmiter:1000_orders:[2,3,4,5,6,7,8,9]_p:[0.5]_samples:1000000_sp:[0.25]_trials:25_data.json"
    with open(data_location+filename,'r') as f:
        accuracy,LT_runtime, LT_klauAccuracy,LT_klauRuntime,vertex_coverage,A_motifs_counts, order_idx, k_idx = process_klau_data(json.load(f))

    filename = "RandomGeometric_degreedist:LogNormal_TabuSearchkvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_p:[0.5]_samples:1000000_sp:[0.25]_trials:25_data.json"
    with open(data_location+filename,'r') as f:
        _, LT_TabuAccuracy, LT_TabuRuntime, _, order_idx, k_idx = process_tabu_data(json.load(f))

    def make_percentile_plot(plot_ax, x_domain,data,color,**kwargs):   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
            (20,80,.2,color)
        ]
        plot_percentiles(plot_ax,  data.T, x_domain, lines, ribbons,**kwargs)

    def make_violin_plot_v2(ax,data,precision=2,c=None,v_alpha=.8,format="default",xlim=None,xscale="linear"):

        med_bbox = dict(boxstyle="round", ec="w", fc="w", alpha=.5,pad=.05)
        bbox = dict(boxstyle="round", ec="w", fc="w", alpha=.5,pad=.01)

        max_elem = np.max(data)
        center = np.min(data)
        scaled_data = [(d-center)/max_elem for d in data]
        if xscale=="linear":
            v = ax.violinplot(data,[.455], points=100, showmeans=False,widths=.08,
                        showextrema=False, showmedians=True,vert=True)#,quantiles=[[.2,.8]]*len(data2))
            ax.set_xlim(0.4, 0.54)
        elif xscale=="log":
            v = ax.violinplot(np.log10(data),[.455], points=100, showmeans=False,widths=.005,
                              showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        else:
            raise ValueError(f"supports xscale values: 'linear' & 'log'; Got {xscale}.")


        #  --  update median lines to have a gap  --  #
        ((x0,y0),(x1,y1)) = v["cmedians"].get_segments()[0]

        newMedianLines = [[(0.415,y1),(.43,y1)]]
        v["cmedians"].set_segments(newMedianLines)

        # -- write data values as text
        tick_ypos = .1
        def format_string(val):
            if np.abs(val-1.0) < 1e-2:
                return f"{val:.{1}f}" 
            elif val >= 1000: 
                return f"{round(val/1000,1)}k"
            else: 
                return f"{val:.{precision}f}".strip("0")

        #format_string = lambda val: f"{val:.{1}f}" if np.abs(val-1.0) <1e-2 else f"{val:.{precision}f}".strip("0")
        xloc=.4
        ax.annotate(format_string(np.median(data)),xy=(xloc,.4),xycoords="axes fraction",ha="left",va="center",fontsize=10)#.set_bbox(med_bbox)
        #ax.annotate(format_string(np.min(data)),xy=(.025,.1),xycoords="axes fraction",ha="left",fontsize=6,alpha=.8)#.set_bbox(bbox)
        #ax.annotate(format_string(np.max(data)),xy=(.925,.8),xycoords="axes fraction",ha="right",fontsize=6,alpha=.8)#.set_bbox(bbox)

        ax.annotate(format_string(np.max(data)),xy=(xloc,.8),xycoords="axes fraction",ha="left",va="center",fontsize=8,alpha=.8)#.set_bbox(bbox)
        ax.annotate(format_string(np.min(data)),xy=(xloc,.0),xycoords="axes fraction",ha="left",va="center",fontsize=8,alpha=.8)#.set_bbox(bbox)


        if c is not None:
            v["cmedians"].set_color(c)
            v["cmedians"].set_alpha(.75)
            for b in v['bodies']:
                # set colors 
                b.set_facecolor(c) #b.set_facecolor("None")
                b.set_edgecolor("None") #b.set_edgecolor(c)            
                b.set_alpha(v_alpha)
                #b.set_color(c)

                #  -- only plot the left half of violin plot --  #
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf,m)
                
                #  -- clip the top of the med-line to the top of the violin plot --  #
                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    #new_max_y += .04
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                #clip_to_top_of_violin(v["cmedians"])
 

    def make_violin_plot_legend(ax,c="k"):
        
        np.random.seed(12)#Ok looking:12
        v = ax.violinplot([np.random.normal() for i in range(50)], points=100, positions=[.1], showmeans=False, 
                        showextrema=False, showmedians=True,vert=True,widths=[.5])#,quantiles=[[.2,.8]]*len(data2))
        #ax.axes.set_axis_off()
        #ax.set_ylim(.5,1.0)
        #  --  update median lines to have a gap  --  #
        ((x0,y0),(x1,y1)) = v["cmedians"].get_segments()[0]
        #newMedianLines = [[(x0,y0-.125),(x0,y0 + (y1-y0)/2 -.1)]]#,[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]

        newMedianLines = [[(x1-2.5,y1),(x1-.3,y1)]]
        v["cmedians"].set_segments(newMedianLines)

        xloc = .6

        ax.annotate(f"min ",xy=(xloc,0.2),xycoords="axes fraction",ha="left",va="center",fontsize=11,alpha=.8)
        ax.annotate("med.",xy=(xloc,.475),xycoords="axes fraction",ha="left",va="center",fontsize=14)
        ax.annotate(f"max ",xy=(xloc,.75),xycoords="axes fraction",ha="left",va="center",fontsize=11,alpha=.8)

        if c is not None:
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor("None")
                b.set_alpha(.3)
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0],-np.inf,m)

                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    new_max_y += .02
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                #clip_to_top_of_violin(v["cmedians"])


    #  -- add in order labels overhead --  #

    global_ax.set_yticklabels([])
    global_ax.set_yticklabels([])


    widths = [.7,.3]
    #spec = fig.add_gridspec(nrows=5,,hspace=0.2,wspace=0.1,height_ratios=widths,left=.2,right=1.075,top=.95,bottom=.05)
    global_spec = fig.add_gridspec(nrows=2, ncols=1,
                          left=.2,right=1.075,top=.95,bottom=.05,
                          wspace=0.0,hspace=0.1,
                          height_ratios=widths
                          )
    widths = [.5, 3, 2]
    top_gs = global_spec[0].subgridspec(nrows=3,ncols=1+len(order_idx),
                                hspace=0.2,wspace=0.1,
                                height_ratios=widths)
    bottom_gs = global_spec[1].subgridspec(nrows=2,ncols=1+len(order_idx),
                                #left=0.125, right=0.9,top=.95,bottom=.075,
                                wspace=0.1,hspace=0.4)



    allCAx = []
    allAccAx = [] 
    allRtAx = [] 
    allVCAx = []
    allMotifCountAx = []
    allSparsityAx = []
    
    all_axes = np.empty((5,len(order_idx)),object)
    first = True
    annotation_idx = 1

    if filename == "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json":
        k_tick_idx = 0
    else:
        k_tick_idx = 0
    parity = 0
    for idx,(order,i) in enumerate(order_idx.items()):
        #
        #  Clique size
        #
        ax = fig.add_subplot(top_gs[0,i])
        all_axes[0,i] = ax
        allCAx.append(ax)
        ax.annotate(f"{order}",xy=(.5, .85), xycoords='axes fraction', c="k",size=10,ha="center",va="center",weight="bold")

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        #
        #  Accuracy Plots
        #
        if idx == annotation_idx:
            ax = fig.add_subplot(top_gs[1,i],zorder=5)#,sharey=allAccAx[0])
        elif idx == k_tick_idx:
            ax = fig.add_subplot(top_gs[1,i],zorder=5)
        else:
            ax = fig.add_subplot(top_gs[1,i])#,sharey=allAccAx[0])
        all_axes[1,i] = ax


        allAccAx.append(ax)

        #if idx % 2 != parity:
        #    ax.patch.set_facecolor('k')
        #    ax.patch.set_alpha(0.1)
        
        ax.set_yticks([0.0,0.25,.5,.75,1.0])
        ax.set_ylim(0,1.0)

                
        plt.axhline(y=np.median(accuracy[i,:,:]), color=LT_color,linestyle="solid")#LT_linestyle)
        plt.axhline(y=np.max(accuracy[i,:,:]), color=LT_color,linestyle="dotted")
        make_percentile_plot(ax,k_idx.keys(),LT_klauAccuracy[i,:,:],LT_Klau_color)#,linestyle=LT_Klau_linestyle)
        make_percentile_plot(ax,k_idx.keys(),LT_TabuAccuracy[i,:,:],LT_Tabu_color)#,linestyle=LT_Tabu_linestyle)

        ax.set_xticklabels([])
        ax.set_yticklabels([])


        #
        #  Runtime Plots
        #
        """
        if idx == k_tick_idx:
            ax = fig.add_subplot(spec[2,i],zorder=3)
        else:

        """
        ax = fig.add_subplot(top_gs[2,i])
        all_axes[2,i] = ax
        allRtAx.append(ax)
        

        if filename == "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json":
            ax.set_ylim(np.min(LT_klauRuntime[:,:,:]),600)
        elif filename == "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json":
            ax.set_ylim(np.min(LT_klauRuntime[:,:,:]),1200)
        else:
            ax.set_ylim(np.min(LT_klauRuntime[:,:,:]),2100)
        #ax.set_yscale("log")
        if idx != 0:
            ax.set_yticklabels([])

        ax.set_xticks([15,45,90])

        ax.set_xticklabels([])
            
        make_percentile_plot(ax,k_idx.keys(),LT_klauRuntime[i,:,:],LT_Klau_color)#,linestyle=LT_Klau_linestyle)
        make_percentile_plot(ax,k_idx.keys(),LT_TabuRuntime[i,:,:],LT_Tabu_color)#,linestyle=LT_Tabu_linestyle)

        plt.axhline(y=np.median(LT_runtime[i,:,:]), color=LT_color,linestyle="solid")#LT_linestyle)


        #
        #  Vertex Coverage
        #
        if i == 0:
            ax = fig.add_subplot(bottom_gs[0,i])
        else:
            ax = fig.add_subplot(bottom_gs[0,i],sharey=all_axes[3,0])
        #ax = fig.add_subplot(spec[3,i])
        all_axes[3,i] = ax

        allVCAx.append(ax)
        make_violin_plot_v2(ax,vertex_coverage[order],c="k",v_alpha=.2)


        #
        #  Motif Counts
        #
        '''
        if i == 0: 
            ax = fig.add_subplot(spec[4,i])
        else:
            ax = fig.add_subplot(spec[4,i],sharey=allMotifCountAx[0])
        '''

        if i == 0:
            ax = fig.add_subplot(bottom_gs[1,i])
        else:
            ax = fig.add_subplot(bottom_gs[1,i],sharey=all_axes[4,0])
        all_axes[4,i] = ax
        allMotifCountAx.append(ax)

        make_violin_plot_v2(ax,A_motifs_counts[order],c="k",v_alpha=.2, precision=0)#,xscale="log")

    violinLegendAx = allVCAx[0].inset_axes([-2.1,-1.75,.5,1.5])

    #for ax in chain(allAccAx,allRtAx,allCAx,allVCAx,allMotifCountAx,allSparsityAx,[global_ax],[violinLegendAx]):
    for ax in chain(all_axes.reshape(-1),[global_ax],[violinLegendAx]):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        ax.set_xticklabels([])
        ax.set_yticklabels([])


    for ax in all_axes[1:3,:].reshape(-1):
        ax.xaxis.set_ticks_position('top')
        ax.grid(True,axis='y',linewidth=.5)
        ax.tick_params(axis="x",direction="in", pad=-15)
        ax.set_xticks([15,45,90])
        ax.set_xlim(min(k_idx.keys()),max(k_idx.keys()))
        
    for ax in all_axes[3:,:].reshape(-1):
        ax.patch.set_facecolor('none')



    for i,ax in enumerate(all_axes[2,:]):

        ax.set_yscale("log")

        ax.set_yticks([1e-1,1e0,1e1,1e2,1e3,1e4])
        ax.set_ylim(1e-1,1e4)

        if i == 0:
            ax.set_yticklabels([r"$10^{-1}$",r"$10^0$",r"$10^1$",r"$10^2$",r"$10^3$",None],ha="left")
            ax.tick_params(axis="y",direction="out",which='both', length=7.5,pad=20)
        else:
            ax.set_yticklabels([])
      
        y_gridlines = ax.yaxis.get_gridlines()
        for i in [0,-1]:
            if i == 0:
                y_gridlines[i].set_linewidth(1.5*y_gridlines[i].get_linewidth())
            if i == -1:
                y_gridlines[i].set_color("k")


    for ax in allAccAx:
        #ax.xaxis.set_ticks_position('top')
        y_gridlines = ax.yaxis.get_gridlines()
        for i in [0,-1]:
            y_gridlines[i].set_linewidth(1.5*y_gridlines[i].get_linewidth())
            if i == -1:
                y_gridlines[i].set_color("k")




    #  - make violin plot legend -  #
    make_violin_plot_legend(violinLegendAx)



    #  -- Alternate tick labels to opposite axes --  #

    label_xpos = -.1

    allCAx[0].annotate("Clique Size",xy=(label_xpos,.85),ha="right",va="center",xycoords='axes fraction')

    #allAccAx[-1].yaxis.set_ticks_position('right')
    allAccAx[0].set_yticklabels([0,.25,.5,.75,1.0])
    allAccAx[0].tick_params(axis="y",direction="out",which='both', length=7.5)
    #allAccAx[0].set_ylabel("Accuracy",rotation=90,labelpad=0,ha="right")
    allAccAx[0].annotate("Accuracy",xy=(-1.5,.5),rotation=0,ha="center",va="center",xycoords='axes fraction')
    
    #allRtAx[-1].yaxis.set_label_position("right")

    allRtAx[0].annotate("Runtime\n(seconds)",xy=(-1.5,.5),ha="center",va="center",xycoords='axes fraction',rotation=0)



    allVCAx[0].annotate("Vertex\nCoverage",xy=(label_xpos,.5),ha="right",va="center",xycoords='axes fraction')
    #  -- Annotate accuracy plots --  #
    allAccAx[0].annotate(r"$\Lambda$T"+"-Klau", xy=(.5, .675), xycoords='axes fraction', c=LT_Klau_color,size=10,ha="center",rotation=20)
    allAccAx[0].annotate(r"$\Lambda$T"+"-LS", xy=(.575, .3), xycoords='axes fraction', c=LT_Tabu_color,size=10,ha="center",zorder=4)
    allAccAx[annotation_idx].annotate("maximum "+r"$\Lambda$"+"T", xy=(.025, .375), xycoords='axes fraction', c=LT_color,size=10,ha="left")
    allAccAx[annotation_idx].annotate("median "+r"$\Lambda$"+"T", xy=(.025, .1), xycoords='axes fraction', c=LT_color,size=10,ha="left")

    allMotifCountAx[0].annotate("Motifs\nin A",xy=(label_xpos,.5),ha="right",va="center",xycoords='axes fraction')


    #  -- Add in x-domain annotations mid plot --  #

    allAccAx[k_tick_idx].xaxis.set_label_position("top")
    allAccAx[k_tick_idx].xaxis.set_ticks_position('top')
    allAccAx[k_tick_idx].tick_params(axis="x",direction="in", pad=.1)
    allAccAx[k_tick_idx].annotate("nearest\nneighbors ("+r"$K$"+')',xy=(0.0,-.125),ha="left",va="bottom",xycoords='axes fraction',zorder=10,fontsize=8)

    allAccAx[k_tick_idx].set_xticklabels([15,45,90],zorder=3)
    #allAccAx[k_tick_idx].grid(zorder=0)




    #plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def LambdaTAME_increasing_clique_size_summarized(save_path=None):

    plt.rc('text.latex', preamble=r'\usepackage{/Users/charlie/Documents/Code/TKPExperimentPlots/latex/dgleich-math}')
    
    5,3
    fig = plt.figure(figsize=(6,5))
    global_ax = plt.gca()

    #
    #  Subroutines
    #


    def underline_text(ax,text,c,linestyle):
        tb = text.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        ax.annotate('', xy=(tb.xmin,tb.y0), xytext=(tb.xmax,tb.y0),             xycoords="figure fraction",arrowprops=dict(arrowstyle="-", color=c,linestyle=linestyle,linewidth=1.5,alpha=.8))

    def mark_as_algorithm(ax,text,c,linestyle,algorithm="Klau"):
        
        tb = text.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        
        # calculate asymmetry of x and y axes:
        x0, y0 = fig.transFigure.transform((0, 0)) # lower left in pixels
        x1, y1 = fig.transFigure.transform((1, 1)) # upper right in pixes
        dx = x1 - x0
        dy = y1 - y0
        maxd = max(dx, dy)

        #ax.annotate('', xy=(tb.xmin,tb.y0), xytext=(tb.xmax,tb.y0),             xycoords="figure fraction",arrowprops=dict(arrowstyle="-", color=LRT_color,linestyle=LRT_linestyle,linewidth=1.5,alpha=.8))
        """
        def algorithm_patches(algo):
            if algo == "LRTAME":
                radius=.02
                height = radius * maxd / dy
                width = radius * maxd / dx

                return patches.Ellipse((tb.xmin-.015,tb.y0+(5/8)*tb.height),width, height,color=LRT_color,transform=fig.transFigure)
                
            elif algo == "TAME":
                side_length=.015
                height = side_length * maxd / dy
                width = side_length * maxd / dx

                return patches.Rectangle((tb.xmax+.01,tb.y0+.5*(tb.height - side_length)),
                                            width, height,color=T_color,
                                            transform=fig.transFigure)
            else:
                raise ValueError(f"algorithm must be either 'TAME' or 'LRTAME', got {algo}.\n")
        """

        if algorithm == "Klau":
                radius=.01
                height = radius * maxd / dy
                width = radius * maxd / dx

                p=ax.add_patch(patches.Ellipse((tb.xmin-.015,tb.y0+(5/8)*tb.height),width, height,color=LT_Klau_color,transform=fig.transFigure,clip_on=False))
                
        elif algorithm == "Tabu":
            side_length=.0075
            height = side_length * maxd / dy
            width = side_length * maxd / dx

            p=ax.add_patch(patches.Rectangle((tb.xmax+.01,tb.y0+.5*(tb.height - side_length)),
                                        width, height,color=LT_Tabu_color,
                                        transform=fig.transFigure,clip_on=False))
        else:
            raise ValueError(f"algorithm must be either 'Klau' or 'Tabu', got {algorithm}.\n")
    

        #p=ax.add_patch(algorithm_patches(algorithm))
        #ax.add_patch(patches.Ellipse((tb.xmin-.015,tb.y0+tb.height/2),width, height,color=LRT_color,transform=fig.transFigure))
        #ax.add_patch(patches.Ellipse((tb.xmax,tb.y0),width, height,color=LRT_color,#transform=fig.transFigure),)
        """
        #width = .05
        #height = .01
        xshift = tb.width*.2
        height = .03
        width = .0075
        ax.add_patch(patches.Rectangle((tb.xmin-xshift,tb.y0 + (3/4)*tb.height),width, height,color=LRT_color,transform=fig.transFigure))

        height = .015 #* maxd / dy
        width = .02 #* maxd / dx
        xshift = tb.width*.1
        ax.add_patch(patches.Rectangle((tb.xmin-xshift,tb.y1), width,height,color=LRT_color,transform=fig.transFigure))
        """

    extremal_tick_ypos = .1

    def make_violin_plot(ax,data,precision=2,c=None,v_alpha=.8,format="default",xlim=None,xscale="linear",column_type=None):


        #background_v = ax.violinplot(data, points=100, positions=[0.5], showmeans=False, 
        #                showextrema=False, showmedians=False,widths=.5,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #
        #positions=[0.5], ,widths=.5
        if xscale=="linear":
            v = ax.violinplot(data,[.5], points=100, showmeans=False,widths=.15,
                        showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        elif xscale=="log":
            v = ax.violinplot(np.log10(data), points=100, showmeans=False,widths=.15,showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        else:
            raise ValueError(f"supports xscale values: 'linear' & 'log'; Got {xscale}.")
        #ax.set_ylim(0.95, 1.3)
        #ax.set_xlim(np.min(data),np.max(data))

    

        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]

        newMedianLines = [[(x0,y1),(x0,y1+.7)]]
        v["cmedians"].set_segments(newMedianLines)

        # -- place extremal markers underneath
        """
        v['cbars'].set_segments([]) # turns off x-axis spine
        for segment in [v["cmaxes"],v["cmins"]]:
            ((x,y0),(_,y1)) = segment.get_segments()[0]
            segment.set_segments([[(x,0.45),(x,.525)]])
            segment.set_color(c)
        """

        # -- write data values as text
        bbox = dict(boxstyle="round", ec="w", fc="w", alpha=.5,pad=.05)
        extremal_tick_ypos = .1
        if column_type is None:

            if format == "default":
                ax.annotate(f"{np.median(data):.{precision}f}",xy=(.5,.4),xycoords="axes fraction",ha="center",fontsize=10).set_bbox(bbox)
                ax.annotate(f"{np.min(data):.{precision}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
                ax.annotate(f"{np.max(data):.{precision}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
            elif format == "scientific":
                ax.annotate(f"{np.median(data):.{precision}e}",xy=(.5,.4),xycoords="axes fraction",ha="center",fontsize=10).set_bbox(bbox)
                ax.annotate(f"{np.min(data):.{precision}e}",xy=(.025,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
                ax.annotate(f"{np.max(data):.{precision}e}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
            else:
                print(f"expecting format to be either 'default' or 'scientific', got:{format}")
        elif column_type == "merged_axis":
            pass
        else:
            raise ValueError("column_type expecting 'merged_axis' or None, but got {column_type}\n")

        if c is not None:
            v["cmedians"].set_color(c)
            v["cmedians"].set_alpha(.75)
            for b in v['bodies']:
                # set colors 
                b.set_facecolor("None")
                b.set_edgecolor(c)            
                b.set_alpha(v_alpha)
                #b.set_color(c)

                #  -- only plot the top half of violin plot --  #
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)
                
                #  -- clip the top of the med-line to the top of the violin plot --  #
                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    #new_max_y += .04
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)

                #clip_to_top_of_violin(v["cmaxes"])
                #clip_to_top_of_violin(v["cmins"])
                clip_to_top_of_violin(v["cmedians"])
 
    def make_violin_plot_merged_axis(ax,data1,data2,c1,c2,marker1,marker2,format=None,**kwargs):


        make_violin_plot(ax,data1,**dict(kwargs,c=c1,column_type="merged_axis"))
        make_violin_plot(ax,data2,**dict(kwargs,format=format,c=c2,column_type="merged_axis"))
        #ax.scatter(np.median(data1),.65,marker=marker1,s=5)

        marker_size = 12.5
        print(ax.get_ylim())
        x,marker_y_loc = ax.get_ylim()
        ax.set_ylim(x,marker_y_loc + .05)
        ax.scatter(np.median(data1),marker_y_loc,marker=LT_Klau_marker,color=LT_Klau_color,s=marker_size)
        ax.scatter(np.median(data2),marker_y_loc,marker=LT_Tabu_marker,color=LT_Tabu_color,s=marker_size)



        min1 = np.min(data1)   
        min2 = np.min(data2)

        if min1 < min2:
            #text = f"{min1:.{kwargs['precision']}f}"
            #underlined_annotation(fig,ax,(.075,extremal_tick_ypos),text,linestyle=LRT_linestyle,ha="left",fontsize=8,alpha=.8)

            text = ax.annotate(f"{min1:.{kwargs['precision']}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8) 
            mark_as_algorithm(ax,text,T_color,T_linestyle,algorithm="Klau")
            #underline_text(ax,text,T_color,T_linestyle) 
            """
            tb = text.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            ax.annotate('', xy=(tb.xmin,tb.y0), xytext=(tb.xmax,tb.y0),             xycoords="figure fraction",arrowprops=dict(arrowstyle="-", color='k',linestyle=T_linestyle))
            """
        else:
            text = ax.annotate(f"{min2:.{kwargs['precision']}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)  
            mark_as_algorithm(ax,text,LRT_color,LRT_linestyle,algorithm="Tabu")
            #underline_text(ax,text,LRT_color,LRT_linestyle)

        #minimum_val = min([np.min(data1),np.min(data2)])
        maximum_val = min([np.max(data1),np.max(data2)])
        max1 = np.max(data1)   
        max2 = np.max(data2)
        if max1 > max2:
            text = f"{maximum_val:.{kwargs['precision']}f}"
            text = ax.annotate(f"{max1:.{kwargs['precision']}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)  
            mark_as_algorithm(ax,text,T_color,T_linestyle,algorithm="Klau")
            #underline_text(ax,text,T_color,T_linestyle)
            #underlined_annotation(fig,ax,(.925,extremal_tick_ypos),text,linestyle=LRT_linestyle,ha="right",fontsize=8,alpha=.8)
        else:
            text = ax.annotate(f"{maximum_val:.{kwargs['precision']}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
            mark_as_algorithm(ax,text,LRT_color,LRT_linestyle,algorithm="Tabu")
            #underline_text(ax,text,LRT_color,LRT_linestyle)

        bbox = dict(boxstyle="round", ec="w", fc="w", alpha=.5,pad=.025)
        ax.annotate(f"{np.median(data1):.{kwargs['precision']}f}",xy=(.7,.4),xycoords="axes fraction",ha="center",fontsize=10).set_bbox(bbox)
        ax.annotate(f"{np.median(data2):.{kwargs['precision']}f}",xy=(.3,.4),xycoords="axes fraction",ha="center",fontsize=10).set_bbox(bbox)




        #for x in sorted(dir(text)):
        #    print(x)
        """
        if format is None:        
            ax.annotate(f"{np.median(data1):.{kwargs[:precision]}f}",xy=(.5,.2),xycoords="axes fraction",ha="center",fontsize=10)
            ax.annotate(f"{np.min(data1):.{precision}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
            ax.annotate(f"{np.max(data1):.{precision}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
        elif format == "scientific":
            ax.annotate(f"{np.median(data):.{precision}e}",xy=(.5,.2),xycoords="axes fraction",ha="center",fontsize=10)
            ax.annotate(f"{np.min(data):.{precision}e}",xy=(.025,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
            ax.annotate(f"{np.max(data):.{precision}e}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
        else:
            print(f"expecting format to be 'scientific' or None, got:{format}")
        """

    def make_violin_plot_legend(ax,c="k"):
        
        np.random.seed(12)#Ok looking:12
        v = ax.violinplot([np.random.normal() for i in range(50)], points=100, positions=[.6], showmeans=False, 
                        showextrema=False, showmedians=True,widths=.6,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #ax.axes.set_axis_off()
        ax.set_ylim(.5,1.0)
        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        #newMedianLines = [[(x0,y0-.125),(x0,y0 + (y1-y0)/2 -.1)]]#,[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
        newMedianLines = [[(x0,y1 +.05),(x0,1.5)]]
        v["cmedians"].set_segments(newMedianLines)

        ax.annotate("median",xy=(.5,.4),xycoords="axes fraction",ha="center",va="center",fontsize=10)
        ax.annotate(f"min",xy=(.025,-.125),xycoords="axes fraction",ha="left",fontsize=9,alpha=.8)
        ax.annotate(f"max",xy=(.975,-.125),xycoords="axes fraction",ha="right",fontsize=9,alpha=.8)

        if c is not None:
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor(c)
                b.set_alpha(.3)
                b.set_color(c)
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)

                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    new_max_y += .02
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])

    def make_merged_violin_plot_legend(ax):
        
        np.random.seed(12)#Ok looking:12
        v1= ax.violinplot([np.random.normal(-.25,.25) for i in range(50)], points=100, positions=[.6], showmeans=False, showextrema=False, showmedians=True,widths=.6,vert=False)#,quantiles=[[.2,.8]]*len(data2))

        v2 = ax.violinplot([np.random.normal(.5,.25) for i in range(50)], points=100, positions=[.6], 
                          showmeans=False, showextrema=False, showmedians=True,widths=.6,vert=False)
        ax.set_ylim(.5,1.0)

        for (c,v) in [(LRT_color,v1),(T_color,v2)]:
            #  --  update median lines to have a gap  --  #
            ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
            #newMedianLines = [[(x0,y0-.125),(x0,y0 + (y1-y0)/2 -.1)]]#,[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
            newMedianLines = [[(x0,y1 +.05),(x0,1.5)]]
            v["cmedians"].set_segments(newMedianLines)

            med_label1 = ax.annotate(r"$\Lambda$T"+"Klau med.",xy=(.25,.35),xycoords="axes fraction",ha="center",va="center",fontsize=10)
            #mark_as_algorithm(ax,med_label1,LRT_color,LRT_linestyle,algorithm="LRTAME")
            med_label2 = ax.annotate(r"$\Lambda$T"+"Tabu med.",xy=(.7,.35),xycoords="axes fraction",ha="center",va="center",fontsize=10)
            #mark_as_algorithm(ax,med_label2,T_color,T_linestyle,algorithm="TAME")
            min_label = ax.annotate(f"min",xy=(.075,-.125),xycoords="axes fraction",ha="left",fontsize=9,alpha=.8)
            mark_as_algorithm(ax,min_label,LRT_color,LRT_linestyle,algorithm="Klau")
            
            max_label = ax.annotate(f"max",xy=(.925,-.125),xycoords="axes fraction",ha="right",fontsize=9,alpha=.8)
            mark_as_algorithm(ax,max_label,T_color,T_linestyle,algorithm="Tabu")
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor(c)
                b.set_alpha(.3)
                b.set_color(c)
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)

                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    new_max_y += .02
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])


    data_location = TAME_RESULTS + "RG_DupNoise/"
    def process_klau_data(data,version="LT"):
        order_idx = {order:i for (i,order) in enumerate(sorted(set([datum[0] for datum in data])))}
        k_idx = {k:i for (i,k) in enumerate(sorted(set([datum[1] for datum in data])))}
        trials =len(data[0][-1])

        runtime = np.zeros((len(order_idx),len(k_idx),trials))
        accuracy = np.zeros((len(order_idx),len(k_idx),trials))
        postProcessingAccuracy = np.zeros((len(order_idx),len(k_idx),trials))
        triMatch = np.zeros((len(order_idx),len(k_idx),trials))
        postProcessingRuntime = np.zeros((len(order_idx),len(k_idx),trials))
        #sparsity = np.zeros((len(order_idx),len(k_idx),trials))
        trial_idx = np.zeros((len(order_idx),len(k_idx)),int)

        vertex_coverage = {}
        A_motifs_counts = {}


        for (order,k,results) in data:
            # expecting results:
            #     (seed,p, n, sp, acc, dup_tol_acc, matched_matching_score, A_motifs, B_motifs, A_motifDistribution, B_motifsDistribution,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt)
            i = order_idx[order]
            j = k_idx[k]
            print(f"p:{results[0][1]} n:{results[0][2]} sp:{results[0][3]}")
    
            accuracy[i,j,:] = [x[4] for x in results]
            if version=="LT":
                runtime[i,j,:] = [sum([sum(val) for val in result[11].values()]) for result in results] 
                postProcessingAccuracy[i,j,:] = [x[15] for x in results]
                postProcessingRuntime[i,j,:] = [x[17] + x[18] for x in results]
                     #sparsity[i,j,:] = [x[19] for x in results]
                if k == min(k_idx.keys()): #only do this once over k 

                    #determine the size of A vs. B motif data
                    if len(results[0][9]) == results[0][2]:
                        vertex_coverage[order] = [len(list(filter(lambda y: y!= 0.0,x[9][0])))/x[2] for x in results]
                        A_motifs_counts[order]= [x[7][0] for x in results]
                    else:
                        vertex_coverage[order] = [len(list(filter(lambda y: y!= 0.0,x[10][0])))/x[2] for x in results]
                        A_motifs_counts[order]= [x[8][0] for x in results]


            elif version=="LRT":
                postProcessingAccuracy[i,j,:] = [x[14] for x in results]
                postProcessingRuntime[i,j,:] = [x[16] + x[17] for x in results]

        if version=="LT":
            return accuracy,runtime, postProcessingAccuracy, postProcessingRuntime,vertex_coverage,A_motifs_counts, order_idx, k_idx #triMatch,LT_klauTriMatch, p_idx, n_idx 
        else:
            return accuracy, postProcessingAccuracy, postProcessingRuntime,order_idx, k_idx #triMatch,LT_klauTriMatch, p_idx, n_idx 
    
    def process_tabu_data(data,version="LT"):
        order_idx = {order:i for (i,order) in enumerate(sorted(set([datum[0] for datum in data])))}
        k_idx = {k:i for (i,k) in enumerate(sorted(set([datum[1] for datum in data])))}
        trials =len(data[0][-1])

        accuracy = np.zeros((len(order_idx),len(k_idx),trials))
        tabuAccuracy = np.zeros((len(order_idx),len(k_idx),trials))
        triMatch = np.zeros((len(order_idx),len(k_idx),trials))
        tabuRuntime = np.zeros((len(order_idx),len(k_idx),trials))
        #sparsity = np.zeros((len(order_idx),len(k_idx),trials))
        trial_idx = np.zeros((len(order_idx),len(k_idx)),int)

        vertex_coverage = {}



        for (order,k,results) in data:
            # expecting results:
            #     (seed,p, n, sp, acc, dup_tol_acc, matched_matching_score, A_motifs, B_motifs, A_motifDistribution, B_motifsDistribution,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt)
            i = order_idx[order]
            j = k_idx[k]
            accuracy[i,j,:] = [x[4] for x in results]
            if version == "LT":
                tabuAccuracy[i,j,:] = [x[15] for x in results]
                tabuRuntime[i,j,:] = [x[18] for x in results]
                #sparsity[i,j,:] = [x[19] for x in results]
                if k == min(k_idx.keys()): #only do this once over k 
                    vertex_coverage[order] = [len(list(filter(lambda y: y!= 0.0,x[9][0])))/x[2] for x in results]
                    

            elif version == "LRT":
                print(i," ",j)
                tabuAccuracy[i,j,:] = [x[14] for x in results]
                tabuRuntime[i,j,:] = [x[17] for x in results]

        if version == "LT":
            return accuracy, tabuAccuracy,tabuRuntime,vertex_coverage, order_idx, k_idx
        else:
            return accuracy, tabuAccuracy,tabuRuntime,order_idx, k_idx
    
    #
    #   Parse the Data  
    #


    #filename = "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json"
    filename = "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_KAmiter:1000_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json" # exp uses sp:10
    filename = "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_KAmiter:100_orders:[2,3,4,5,6,7,8,9]_p:[0.5]_samples:1000000_sp:[0.25]_trials:25_data.json"
    filename = "RandomGeometric_degreedist:LogNormal_KlauAlgokvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_KAmiter:1000_orders:[2,3,4,5,6,7,8,9]_p:[0.5]_samples:1000000_sp:[0.25]_trials:25_data.json"
    #filename = "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json"
    print("loading Klau Data")
    with open(data_location+filename,'r') as f:
 #   with open(data_location+filename,"r") as f:
        accuracy,LT_runtime, LT_klauAccuracy,LT_klauRuntime,vertex_coverage,A_motifs_counts, order_idx, k_idx = process_klau_data(json.load(f))

    filename = "RandomGeometric_degreedist:LogNormal_TabuSearchkvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json"
    filename = "RandomGeometric_degreedist:LogNormal_TabuSearchkvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_p:[0.5]_samples:1000000_sp:[0.25]_trials:25_data.json"
    print("loading Local Search Data")
    with open(data_location+filename,'r') as f:
        _, LT_TabuAccuracy, LT_TabuRuntime, _, order_idx, k_idx = process_tabu_data(json.load(f))

    """
    filename = "RandomGeometric_LRTAME_degreedist:LogNormal_KlauAlgokvals:[15,30,45,60,75,90]_n:[500]_noiseModel:Duplication_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json"
    with open(data_location+filename,'r') as f:
        _, LRT_klauAccuracy, LRT_klauRuntime, order_idx, k_idx = process_klau_data(json.load(f),"LRT")


    filename = "RandomGeometric_LRTAME_degreedist:LogNormal_TabuSearchkvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json"
    with open(data_location+filename,'r') as f:
        _, LRT_TabuAccuracy, LRT_TabuRuntime, order_idx, k_idx = process_tabu_data(json.load(f),"LRT")
    """

    """ Old Plotting code 
    def make_percentile_plot(plot_ax, x_domain,data,color,**kwargs):
        #TODO: check on scope priorities of ax for closures   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
            (20,80,.2,color)
        ]
        plot_percentiles(plot_ax,  data.T, x_domain, lines, ribbons,**kwargs)


    """


  


    #  -- add in order labels overhead --  #

    global_ax.set_yticklabels([])
    global_ax.set_yticklabels([])

    order_label_idx = 0
    gap1 = 1 
    accuracy_idx = 2 
    vertex_coverage_idx = 3
    motif_count_idx = 4
    gap2 = 5 
    runtime_idx = 6
    

    widths = [1, .25,4,2,2,.25,4]
    # = [1]*len(order_idx)
    spec = fig.add_gridspec(nrows=len(order_idx)+1,ncols=5+2,
                            hspace=0.05,wspace=0.025,
                            left=.025,right=.975,top=.85,bottom=.025,
                            width_ratios=widths)

    allCAx = []
    allAccAx = [] 
    allRtAx = [] 
    allVCAx = []
    allMotifCountAx = []
    allSparsityAx = []
    first = True
    annotation_idx = 4
    k_tick_idx = 4
        
    parity = 0
    for idx,(order,i) in enumerate(order_idx.items()):
        #
        #  Clique size
        #
        ax = fig.add_subplot(spec[i,order_label_idx])
        
        allCAx.append(ax)

        ax.annotate(f"{order}",xy=(.5, .5), xycoords='axes fraction', c="k",size=10,ha="center",va="center")

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax = fig.add_subplot(spec[i,accuracy_idx])#,sharey=allAccAx[0])
        allAccAx.append(ax)

        #if idx % 2 != parity:
        #    ax.patch.set_facecolor('k')
        #    ax.patch.set_alpha(0.1)
        


        #  --  Accuracy Plots  --  #

        #ax.set_yticks([0.0,0.25,.5,.75,1.0])
        #ax.set_ylim(0,1.0)

        #ax.set_xticks([])

        
        
        #plt.axhline(y=np.median(accuracy[i,:,:]), color=LT_color,linestyle=LT_linestyle)
        #plt.axhline(y=np.max(accuracy[i,:,:]), color=LT_color,linestyle="dotted")

        default_k = 30

        make_violin_plot_merged_axis(ax,LT_klauAccuracy[i,k_idx[default_k],:],LT_TabuAccuracy[i,k_idx[default_k],:],LT_Klau_color,LT_Tabu_color,None,None,precision=2)
        #make_violin_plot(ax,LT_klauAccuracy[i,k_idx[default_k],:],precision=2,c=LT_Klau_color)
        #make_violin_plot(ax,LT_TabuAccuracy[i,k_idx[default_k],:],precision=2,c=LT_Tabu_color)

  
        #
        #  Runtime Plots
        #
        
        ax = fig.add_subplot(spec[i,runtime_idx])
        allRtAx.append(ax)
        
        make_violin_plot_merged_axis(ax,LT_klauRuntime[i,k_idx[default_k],:],LT_TabuRuntime[i,k_idx[default_k],:],LT_Klau_color,LT_Tabu_color,None,None,precision=1)
        
        #make_violin_plot(ax,LT_klauRuntime[i,k_idx[default_k],:],precision=2,c=LT_Klau_color)
        #make_violin_plot(ax,LT_TabuRuntime[i,k_idx[default_k],:],precision=2,c=LT_Tabu_color)
   

        #
        #  Vertex Coverage
        #
        ax = fig.add_subplot(spec[i,vertex_coverage_idx])
        
        allVCAx.append(ax)

        
        make_violin_plot(ax,vertex_coverage[order],c="k",v_alpha=.1)


        #
        #  Motif Counts
        #
        
        ax = fig.add_subplot(spec[i,motif_count_idx])
        
        allMotifCountAx.append(ax)
        make_violin_plot(ax,A_motifs_counts[order],c="k",v_alpha=.1,precision=0)





    violinLegendAx = fig.add_subplot(spec[-1,1:3])
    violinMergedAxisLegendAx = fig.add_subplot(spec[-1,3:6])


    for ax in chain(allAccAx,allRtAx,allCAx,allVCAx,allMotifCountAx,allSparsityAx,[global_ax,violinLegendAx,violinMergedAxisLegendAx]):#,[violinLegendAx]):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    """
    for ax in chain(allAccAx,allRtAx,allSparsityAx):
        ax.set_xticks([15,45,90])
        ax.set_xlim(min(k_idx.keys()),max(k_idx.keys()))
    """
    #  - make violin plot legend -  #
    make_violin_plot_legend(violinLegendAx)
    make_merged_violin_plot_legend(violinMergedAxisLegendAx)
    #  -- Add in x-domain annotations mid plot --  #

    #allRtAx[k_tick_idx].xaxis.set_label_position("top")
    #allRtAx[k_tick_idx].xaxis.set_ticks_position('top')
    #allRtAx[k_tick_idx].tick_params(axis="x",direction="out", pad=-15)
    #allAccAx[k_tick_idx].annotate("nearest neighbors",xy=(.5,.05),ha="center",xycoords='axes fraction')

    #allRtAx[k_tick_idx].set_xticklabels([15,45,90],zorder=5)
    #allRtAx[k_tick_idx].set_axisbelow(True)
    #allRtAx[0].set_xlabel("nearest\nneighbors")



    #  -- Alternate tick labels to opposite axes --  #
    title_yloc = 1.25
    allCAx[0].annotate("Clique\nSize",xy=(.5,title_yloc ),ha="center",va="center",xycoords='axes fraction')
    allAccAx[0].annotate("Accuracy",xy=(.5,title_yloc ),ha="center",va="center",xycoords='axes fraction')
    allRtAx[0].annotate("Runtime (s)",xy=(.5,title_yloc ),ha="center",va="center",xycoords='axes fraction')
    allVCAx[0].annotate("Vertex\nCoverage",xy=(.5,title_yloc),ha="center",va="center",xycoords='axes fraction')
    #  -- Annotate accuracy plots --  #
    allMotifCountAx[0].annotate("A Motifs",xy=(.5,title_yloc ),ha="center",va="center",xycoords='axes fraction')

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)





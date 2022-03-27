import os

import numpy as np
import datetime as dt


from statistics import mean 

TAME_CPP_LOG_LOC = "../data/TAME_Experiments/LVGNA_alignments/Cpp_TAME_logs/"

def main():

    return print_best_results_as_latex_table(TAME_CPP_LOG_LOC)
    print("Hello World")


def timestamp(line):
    time,line = line.split("]")

    return dt.datetime.strptime(time[1:],"%Y-%m-%d %H:%M:%S"),line.strip()


def params(experiment_log):
    pass

def parse_results(exp_dir=TAME_CPP_LOG_LOC,graphs_to_ignore=[]):

    #find the ordering of the alignments
    log_folders = [f for f in os.listdir(exp_dir) if ".smat" in f]
    order_counts = {}
    graphs = [f.split("_B:")[0].split("A:")[-1] for f in log_folders]
    
    for graph in graphs:
        if graph in order_counts:
            order_counts[graph] += 1
        else:
            order_counts[graph] = 1
    #find the 0 indexed B graph
    for graph in [f.split("_B:")[-1] for f in log_folders]:
        if graph not in order_counts:
            order_counts[graph] = 0
    
    #map order_counts to indices
    graphs = list(order_counts.keys())
    perm = sorted(range(len(order_counts)),key=lambda i:order_counts[graphs[i]])
    indexing = {k:i for (k,i) in zip(graphs,perm)}
    
    #A smat files are sorted by frequency, B smat files are sorted alphabetically
    log_folders =\
        sorted(log_folders, key=lambda f: -order_counts[f.split("_B:")[0].split("A:")[-1]])

    n = len(order_counts)

    Opt_tri_ratio = np.zeros((n,n))
    Runtimes = np.zeros((n,n))
    impTTV_runtimes = np.zeros((n,n))
    matching_runtimes = np.zeros((n,n))
    post_processing_runtimes = np.zeros((n,n))
    Opt_edge_match = np.zeros((n,n))
    Opt_pp_edge_match = np.zeros((n,n))
    Opt_pp_tri_match = np.zeros((n,n)) 
    
    last_A_file = "first"
    alphas = [.5,1.0]
    betas = [0.0,1.0,10.0,100.0]
    for folder in log_folders:

        graph_A, graph_B = folder.split("_B:")
        graph_A = graph_A.split("A:")[-1]

        if (graph_A in graphs_to_ignore) and (graph_B in graphs_to_ignore):
            continue
            
       
        
        i,j = indexing[graph_A],indexing[graph_B]

        rt,best_accuracy,imp_TTV_rt, matching_rt,pp_rt, best_edge_match, best_pp_edge_match, best_pp_tri_match = evaluate_exps_over_params(exp_dir + folder,alphas,betas)

        impTTV_runtimes[i,j] = imp_TTV_rt
        impTTV_runtimes[j,i] = impTTV_runtimes[i,j]

        matching_runtimes[i,j] = matching_rt
        matching_runtimes[j,i] = matching_runtimes[i,j]
        
        Runtimes[i,j] = rt
        Runtimes[j,i] = Runtimes[i,j]

        post_processing_runtimes[i,j] = pp_rt
        post_processing_runtimes[j,i] = post_processing_runtimes[i,j]
        
        Opt_edge_match[i,j] = best_edge_match
        Opt_edge_match[j,i] = Opt_edge_match[i,j]
        
        Opt_pp_edge_match[i,j] = best_pp_edge_match
        Opt_pp_edge_match[j,i] = Opt_pp_edge_match[i,j]
        
        Opt_pp_tri_match[i,j] = best_pp_tri_match
        Opt_pp_tri_match[j,i] = Opt_pp_tri_match[i,j]

        Opt_tri_ratio[i,j] = best_accuracy#[1][1]
        Opt_tri_ratio[j,i] = Opt_tri_ratio[i,j]

    return Opt_tri_ratio,Opt_edge_match,Opt_pp_edge_match,Opt_pp_tri_match, Runtimes, impTTV_runtimes, matching_runtimes,post_processing_runtimes, indexing


def print_best_results_as_latex_table(*args):

    Opt_tri_ratio,Opt_edge_match,\
    Opt_pp_edge_match,Opt_pp_tri_match,\
    Runtimes, impTTV_runtimes, matching_runtimes,\
    post_processing_runtimes, \
    indexing = parse_results(*args)

    for (graph,i) in sorted(indexing.items(),key=lambda x: x[1]):
        print(graph + " &" + " & ".join([f"{x:.2f}" for x in Opt_tri_ratio[i,:]]) + "\\\\")

    for (graph,i) in sorted(indexing.items(),key=lambda x: x[1]):
        print(graph + " &" + " & ".join([f"{x:.2f}" for x in Runtimes[i,:]]) + "\\\\")

    


def find_best_params(exp_dir):

    log_files = [f for f in os.listdir(exp_dir) if ".smat" in f]
    
    exp_stats = []
    for f in log_files:
        with open(exp_dir + "/"+f,'r') as f_obj:
            try:
              exp_stats.append((f,evaluate_full_exp(f_obj.readlines())))
            except:
              print(f)
              raise

   
    #finds the result for TAME_edges, TAME_tris, PostP_edges, PostP_tris
    return sum(map(lambda x: x[1][1],exp_stats))/len(exp_stats)
    return max(exp_stats,key=lambda x:x[1][3])
#           max(exp_stats,key=lambda x:x[1][3]),\
#           max(exp_stats,key=lambda x:x[1][4]),\
#           max(exp_stats,key=lambda x:x[1][5]),\


def evaluate_exps_over_params(exp_dir,alphas,betas):
    log_files = os.listdir(exp_dir)
    files_to_evaluate = {}
    for log in log_files:
        param_split = log.split("_alpha:")[-1]
        alpha_val,beta_split = param_split.split("_beta:")
        alpha_val = float(alpha_val) #convert to float
        beta_val = float(beta_split.split(".log")[0])

        if (alpha_val in alphas) and (beta_val in betas):
            
            if (alpha_val,beta_val) in files_to_evaluate: #if there are multiple runs for parameters
                #keep the most recently run result
                new_file_maketime = os.path.getmtime(exp_dir + "/" + log)
                prev_file_maketime =  os.path.getmtime(exp_dir + "/"+files_to_evaluate[(alpha_val,beta_val)])
                if new_file_maketime > prev_file_maketime:
                  files_to_evaluate[(alpha_val,beta_val)] = log
            else:
                files_to_evaluate[(alpha_val,beta_val)] = log

    total_runtime = 0.0
    matching_runtime = 0.0
    imp_TTV_runtime = 0.0
    full_post_processing_time = 0.0
    best_accuracy = -1.0

    for log_file in files_to_evaluate.values():
        with open(exp_dir + "/" + log_file,"r") as f:
            lines = f.readlines()

            avg_iter_rt, impTTV_rt, matching_rt,total_rt, post_processing_time,edge_match,tri_accuracy,pp_edge_match,pp_tri_match = evaluate_full_exp(lines)
            matching_runtime += matching_rt
            imp_TTV_runtime += impTTV_rt
            total_runtime += total_rt
            full_post_processing_time += post_processing_time

            if tri_accuracy > best_accuracy:
                best_accuracy = tri_accuracy
                best_edge_match = edge_match
                best_pp_edge_match = pp_edge_match
                best_pp_tri_match = pp_tri_match

        
    if len(files_to_evaluate) < len(alphas)*len(betas):
        print(f"WARNING:{exp_dir} is missing results")
                
    return total_runtime, best_accuracy, imp_TTV_runtime, matching_runtime, full_post_processing_time, best_edge_match, best_pp_edge_match, best_pp_tri_match

        
            
    
    

def evaluate_full_exp(experiment_logs):
    """
       Given the results of an experiment log, passed in as an iterator over the lines in the file, 
       the program parses the file and returns:
         - the average runtime of each iteration
         - the total amount of time spent computing the imp_TTV
         - the total amount of time spent computing the matchings 
         - the full duration of the the iterations of TAME
         - the full duration of the post processing procedure
         - the ratio of edges matched in the final TAME iteration
         - the ratio of triangles matched in the final TAME iteration
         - the ratio of edges matched in the final Post Processing Iteration
         - the ratio of triangles matched in the final Post Processing Iteration
    """
    
    Iteration_indices = \
        [i for i in range(len(experiment_logs)) if "Iteration" in experiment_logs[i]]
    
    exps = \
        [experiment_logs[Iteration_indices[i]:Iteration_indices[i+1]] for i in range(len(Iteration_indices)-1)]

    try:
        Post_exps = [i for i in range(len(exps)) if int(exps[i][0].split("Iteration ")[-1].split(" ")[0]) == 0][0] 
    except:

        for line in experiment_logs:
            print(line)
        print(Iteration_indices)
        for x in exps:
            print(x)
        raise
    TAME_iter = exps[:Post_exps]
    #trim off the line which contain beginning of Post Processing prompt
    end_index = \
        [i for i in range(len(TAME_iter[-1])) if "Post-processing ..." in TAME_iter[-1][i]][0]
    
    Post_processing_header = TAME_iter[-1][end_index:]
    TAME_iter[-1] = TAME_iter[-1][:end_index]
    Post_processing = exps[Post_exps:]

    post_processing_time = post_processing_runtime(Post_processing_header,Post_processing)
    
    #compute runtimes
    runtime_components = [iteration_runtime_components(exp) for exp in TAME_iter]
    imp_TTV_runtime = sum([x[0] for x in runtime_components])
    matching_runtime = sum([x[1] for x in runtime_components])
    average_TAME_runtime = mean([duration(exp) for exp in TAME_iter])
    full_TAME_runtime = sum([duration(exp) for exp in TAME_iter])
#    print(f"TAME_average_runtime:{average_TAME_runtime}")

    #find best alignment
    TAME_results = [TAME_alignment_stats(exp) for exp in TAME_iter]
    TAME_edge_count = [result[0] for result in TAME_results]
    TAME_tri_count = [result[1] for result in TAME_results]
    
#    opt_TAME_map = [i for i in range(Post_exps) if TAME_alignment_stats(TAME_iter[i])[1] == max(TAME_tri_count)][0]
#    print(f"iteration {opt_TAME_map} yields {max(TAME_tri_count)} triangles")

    Post_Processing_results = [Post_Processing_stats(exp) for exp in Post_processing]
    Post_Processing_edge_count = [result[0] for result in Post_Processing_results]
    Post_Processing_tri_count = [result[1] for result in Post_Processing_results]


    #get the initialization counts
    init = experiment_logs[:Iteration_indices[0]]
    max_triangle_match =\
        min([int(y.split(": ")[-1].strip()) for y in [x for x in init if "Triangles in" in x]])
    max_edge_match=\
        min([int(y.split(": ")[-1].strip()) for y in [x for x in init if "Number of edges after pruning:" in x]])
    
    return average_TAME_runtime,\
           imp_TTV_runtime,\
           matching_runtime,\
           full_TAME_runtime,\
           post_processing_time,\
           max(TAME_edge_count)/max_edge_match,\
           max(TAME_tri_count)/max_triangle_match,\
           max(Post_Processing_edge_count)/max_edge_match,\
           max(Post_Processing_tri_count)/max_triangle_match
    
#    average_Post_process_runtime = mean([duration(exp) for exp in Post_processing])
#    print(f"Post_Processing_average_runtime:{average_Post_process_runtime}")


def Post_Processing_stats(exp):
    tris = int(exp[-2].split(" = ")[-1].strip())
    edges = int(exp[-3].split(" = ")[-1].strip())
    return edges, tris
    
def TAME_alignment_stats(exp):
    time,line = timestamp(exp[-2])
    stats = line.split(',')
    edge_count = int(stats[1].split(" = ")[-1])
    tri_count = int(stats[2].split(" = ")[-1])
    return edge_count, tri_count

def post_processing_runtime(header,exps):
    parse_for_time = lambda line: dt.datetime.strptime(line.split("]")[0].split("[")[-1], '%Y-%m-%d %H:%M:%S') 

    full_runtime = (parse_for_time(exps[-1][-1]) - parse_for_time(header[0])).total_seconds()
    return full_runtime

    
def iteration_runtime_components(exp):
    
    impTTV_rt = float([l for l in exp if "dt impTTV" in l][0].split("impTTV = ")[-1].split(" (")[0])
    matching_rt = float([l for l in exp if "matchin_score" in l][0].split("dt =")[-1].strip())
    return impTTV_rt, matching_rt
    

def duration(exp):
    start_time,_ = timestamp(exp[0])
    end_time,_ = timestamp(exp[-1])

    return (end_time - start_time).seconds
    
    
if __name__=="__main__":
    main()

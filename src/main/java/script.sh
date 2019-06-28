#! /bin/bash

# todo make java / python experiments class set 777 permissions
# todo make java use same args are python script

scripts_dir_name=scripts # folder where this script and siblings are located
classifier_names=( CEE ) # list of classifier names to run experiments on
dataset_names=() # list of dataset names to run experiments on; leave empty for population from file
dataset_name_list_file_path="/gpfs/home/vte14wgu/dataset_name_lists/tsc_2019_no_missing.txt" # path to file containing list of dataset names
queues=( sky-ib ) # list of queues to spread jobs over, set only 1 queue to submit all jobs to same queue
dynamic_queueing=false # set to true to find least busy queue for each job submission, otherwise round-robin iterate over queues
java_mem_in_mb=8000 # amount of memory for the jvm
job_mem_in_mb=$((java_mem_in_mb + 256)) # amount of memory for the lsf job
seeds=( 0 ) # list of seeds; leave empty for default 0-29
max_num_pending_jobs=200 # max number of pending jobs before waiting
verbosity=1 # verbosity; larger number == more printouts
sleep_time_on_pend=60s # time to sleep when waiting on pending jobs
estimate_train=true # whether to estimate train set; true or false
overwrite_results=false # whether to overwrite existing results; true or false
seeds_by_datasets=false # true == one array job per dataset, false == one array job per seed
language=java # language of the script
datasets_dir_path=/gpfs/home/vte14wgu/Univariate2018 # path to folder containing datasets
results_dir_path=$(pwd)/results # path to results folder
script_file_path=$(pwd)/jar.jar # path to jar file
experiment_name=exp # experiment name to prepend job names
log_dir_path=$(pwd)/logs # path to log folder

# if dataset names are not predefined
if [ ${#dataset_names[@]} -eq 0 ]; then
	readarray -t dataset_names < $dataset_name_list_file_path # read the dataset names from file
fi

# if seeds are not predefined
if [ ${#seeds[@]} -eq 0 ]; then
    # populate with default 0 - 29
	seeds=()
	for((i=0;i<30;i++)); do
		seeds+=( $i )
	done
fi

num_jobs=${#dataset_names[@]} # one array job per dataset
job_array_size=${#seeds[@]} # one job per seed in array job
if [ "$seeds_by_datasets" = 'true' ]; then # if running in seeds by datasets
	num_jobs=${#seeds[@]} # one array job per seed
	job_array_size=${#dataset_names[@]} # one job per dataset in array job
fi

# make the log folder and set open permissions
mkdir -p $log_dir_path
chmod 777 $log_dir_path
# make the results folder and set open permissions
mkdir -p $results_dir_path
chmod 777 $results_dir_path

# build the job script
XIFS=$IFS # set IFS to space TODO not sure we need this
IFS=' '

job_template="
#! /bin/bash

dataset_names=(${dataset_names[@]})
resample_seeds=(${seeds[@]})

classifier_name=%s"

if [ "$seeds_by_datasets" = 'true' ]; then
	job_template="$job_template
dataset_name_index=\$((\$LSB_JOBINDEX-1))
resample_seed_index=%s"
else
	job_template="$job_template
dataset_name_index=%s
resample_seed_index=\$((\$LSB_JOBINDEX-1))"
fi


job_template="$job_template

dataset_name=\${dataset_names[\$dataset_name_index]}
resample_seed=\${resample_seeds[\$resample_seed_index]}

"

if [ "$language" = 'python' ]; then
	# setup environment path to root project folder
	job_template="$job_template

export PYTHONPATH=$working_dir_path

module add python/anaconda/2019.3/3.7

python"
elif [ "$language" = 'java' ]; then
	job_template="$job_template

module add java

pp=%s

java -Xms${mem_in_mb}M -Xmx${mem_in_mb}M -d64 -Dorg.slf4j.simpleLogger.deaultLogLevel=off -javaagent:/gpfs/home/vte14wgu/SizeOf.jar -jar"
else
	job_template="$job_template
echo"
fi

job_template="$job_template $script_file_path $datasets_dir_path \$dataset_name $experiment_results_dir_path \$resample_seed \$pp %s"

# if estimating train set
 if [ "$estimate_train" = 'true' ]; then
 	job_template="$job_template -gtf=true" # append arg
 fi

# if overwriting results
if [ "$overwrite_results" = 'true' ]; then
    job_template="$job_template --overwrite_results" # append arg
fi

job_template="$job_template

# make log files and change to open permissions
run_log_dir_path=%s
echo placeholder > \$run_log_dir_path/\${LSB_JOBINDEX}-1.err
echo placeholder > \$run_log_dir_path/\${LSB_JOBINDEX}-1.out
chmod 777 \$run_log_dir_path/\${LSB_JOBINDEX}-1.err
chmod 777 \$run_log_dir_path/\${LSB_JOBINDEX}-1.out
"

# for each array jobs
for((i=0;i<$num_jobs;i++)); do
    # for each classifier
    for classifier_name in "${classifier_names[@]}"; do
        # if dynamic queueing
        if [ "$dynamic_queueing" = 'true' ]; then
            # set queue to most free queue
            queue=$(bash $scripts_dir_name/find_shortest_queue.sh "${queues[@]}")
        else
            # otherwise round-robin iterate over queues
            queue=${queues[0]}
            queues=( "${queues[@]:1}" )
            queues+=( $queue )
        fi
        # find number of pending jobs
        num_pending_jobs=$(2>&1 bjobs | awk '{print $3, $4}' | grep "PEND ${queue}" | wc -l)
        # while too many jobs pending
        while [ "${num_pending_jobs}" -ge "${max_num_pending_jobs}" ]
        do
            # too many pending jobs, wait a bit and try after
            echo $num_pending_jobs pending on $queue, more than $max_num_pending_jobs, will retry in $sleep_time_on_pend
            sleep ${sleep_time_on_pend}
            if [ "$dynamic_queueing" = 'true' ]; then
                queue=$(bash $scripts_dir_name/find_shortest_queue.sh "${queues[@]}")
            fi
            # find number of pending jobs
            num_pending_jobs=$(2>&1 bjobs | awk '{print $3, $4}' | grep "PEND ${queue}" | wc -l)
        done

        job_name="${experiment_name}_${dataset_names[$i]}_${seeds[$i]}"

        job_log_dir_path=$log_dir_path
        mkdir -p $job_log_dir_path
        chmod 777 $job_log_dir_path
        job_log_dir_path="$job_log_dir_path/${dataset_names[$i]}"
        mkdir -p $job_log_dir_path
        chmod 777 $job_log_dir_path

        job=$(printf "$job_template" "$classifier_name" "$i" "$pp" "$strategy" "$job_log_dir_path")

        # echo "$job"
        # exit 1

        if [ "$seeds_by_datasets" = 'false' ]; then # todo try -1 wrap
            bsub -q $queue -oo "$job_log_dir_path/%I.out" -eo "$job_log_dir_path/%I.err" -R \"rusage[mem=$job_mem_in_mb]\" -J "${job_name}_[1-$job_array_size]" -M $job_mem_in_mb "$job"
        else
            bsub -q $queue -oo "$job_log_dir_path/%I.out" -eo "$job_log_dir_path/%I.err" -R \"rusage[mem=$job_mem_in_mb]\" -J "${job_name}_[1]" -M $job_mem_in_mb "$job"
        fi


	done
done

IFS=$XIFS

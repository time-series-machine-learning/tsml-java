package tsml.classifiers.distance_based.utils.experiment;

public class ExperimentRunner {
    public static void main(String[] args) throws Exception {
//        Experiment.main(
//                "-r", "results"
//                , "-d", "/bench/phd/data/all_2019"
//                , "-p", "GunPoint"
//                , "-c", "PF_R5"
//                , "-s", "0"
//                , "-l", "all"
//                , "-o"
//        );

        String cmd =
                " -d /bench/phd/data/all_2019 -p Phoneme -r results -c PF_R5_OOB -s 0 -m 8000 -t 1  -l ALL -e --cp --rcp";
        cmd = cmd.trim();
        args = cmd.split(" ");
        Experiment.main(args);
        
        
//        Experiment.main(
//                "-r", "results"
//                , "-d", "/bench/phd/data/all"
//                , "-p", "GunPoint"
//                , "-c", "PF_R5_OOB"
////                , "-c", "PF_R5"
//                , "-s", "0"
//                , "-l", "all"
//                , "-o"
//                , "--cp"
//                , "--cpi", "30s"
//                , "--rcp"
//                , "-e"
////                , "--ttl", "10s"
////                , "--ttl", "20s"
////                , "--ttl", "30s"
//                , "--ttl", "1m"
//                , "--ttl", "2m"
//                , "--ttl", "3m"
//                , "--ttl", "4m"
//        );
    }
}

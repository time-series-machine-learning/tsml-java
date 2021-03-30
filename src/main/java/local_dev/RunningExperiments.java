package local_dev;

import experiments.Experiments;
import experiments.Experiments.ExperimentalArguments;

import java.util.concurrent.TimeUnit;

public class RunningExperiments {

    public static void main(String[] args) throws Exception{

        String[] classifierNames = {
                "TDE-OOB",
                "STC-OOB",
                "DrCIF-OOB",
                "PF-OOB",
                "Arsenal-OOB"
        };

        String resultsPath = "results/";

        for(String classifierName: classifierNames) {
            System.out.println("Running " + classifierName);
            ExperimentalArguments exp = new ExperimentalArguments();
            exp.dataReadLocation = "src/main/java/experiments/data/tsc/";
            exp.resultsWriteLocation = resultsPath;
            exp.generateErrorEstimateOnTrainSet = true;
            exp.classifierName = classifierName;
            exp.datasetName = "ItalyPowerDemand";
            exp.foldId = 0;
            exp.trainEstimateMethod = "OOB";
            exp.contractTrainTimeNanos = TimeUnit.NANOSECONDS.convert(10, TimeUnit.SECONDS);

            Experiments.setupAndRunExperiment(exp);
        }
    }
}

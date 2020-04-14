package tsml.classifiers.distance_based.utils.experiments;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ExperimentRunner {

    private static final String A = "a";
    @Parameter(names = {A})
    private String hello = "not set";

    public static void main(String[] args) throws Exception {
        ExperimentBatch.main(
            "-r", "results",
            "-c", "ED_1NN",
            "-d", "GunPoint",
            "--dd", "/bench/datasets",
            "-s", "0",
            "-e",
            "--ttc", "5 minutes"
        );
//        ExperimentRunner c = new ExperimentRunner();
//        JCommander.newBuilder().addObject(c).build().parse("a", "b");
//        System.out.println(c.hello);
    }
}

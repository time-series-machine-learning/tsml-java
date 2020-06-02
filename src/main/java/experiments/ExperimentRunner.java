package experiments;

public class ExperimentRunner {

    public static void main(String[] args) throws Exception {
        Thread.sleep(6000);
        Experiments.main(new String[] {
            "-rp=results",
            "-dp=/bench/phd/datasets/uni2018",
            "-f=1",
            "-gtf=false",
            "-l=ALL",
            "--force=true",
            "-dn=BeetleFly",
//            "-cn=PF_R5",
            "-cn=ORIG_PF",
        });
    }
}

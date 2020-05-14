package experiments;

public class ExperimentRunner {

    public static void main(String[] args) throws Exception {
        Experiments.main(new String[] {
            "-rp=results",
            "-dp=/bench/datasets/",
            "-f=1",
            "-gtf=false",
            "-dn=GunPoint",
            "-cn=PF_100",
        });
    }
}

package timeseriesweka.filters;;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

import static experiments.data.DatasetLoading.loadDataNullable;

public class MFCC extends SimpleBatchFilter {
    int nfft = 256;
    Fast_FFT fft;

    public MFCC(){
        fft = new Fast_FFT();
    }

    @Override
    public String globalInfo() {
        return null;
    }

    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
        return null;
    }

    @Override
    public Instances process(Instances instances) throws Exception {
        return null;
    }

    public static void main(String[] args) {
        MFCC mfcc = new MFCC();
        Instances[] data = new Instances[2];
        data[0] = loadDataNullable("D:\\Test\\Datasets\\InsectWingbeat_100Instances\\InsectWingbeat_100Instances");
        //data = resampleInstances(data[0], 1, 0.5);
        System.out.println(data[0].get(0).toString());
        try {
            data[0] = mfcc.process(data[0]);
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println(data[0].get(0).toString());
    }
}

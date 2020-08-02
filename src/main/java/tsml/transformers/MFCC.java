package tsml.transformers;

import static experiments.data.DatasetLoading.loadDataNullable;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.math3.transform.DctNormalization;
import org.apache.commons.math3.transform.FastCosineTransformer;
import org.apache.commons.math3.transform.TransformType;

import tsml.data_containers.TimeSeriesInstance;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class MFCC implements Transformer {

    // Default values (samples), assuming no sample rate set.
    int windowLength = 50;
    int overlapLength = 25;
    // Default values (miliseconds), determined using sample rate.
    int windowDuration = 25;
    int overlapDuration = 15;

    // Check whether there is an attribute called samplerate and use that before
    // deleting it.
    Boolean checkForSampleRate = true;
    int nfft = 512;
    int sampleRate = 16000;
    Spectrogram spectrogram;
    FastCosineTransformer dct = new FastCosineTransformer(DctNormalization.STANDARD_DCT_I);
    int numFilterBanks = 65;
    double[][] filterBank = null;
    double[][] melFreqCepsCo = null;
    // Upper and lower frequencies the filter bank will be applied to (Freq. outside
    // of these will not contribute to overall output.).
    int lowerFreq = 0;
    int upperFreq = 2000;

    public int interval = 0;

    public MFCC() {
        spectrogram = new Spectrogram(windowLength, overlapLength, nfft);
    }

    public void setSampleRate(int sampleRate) {
        this.sampleRate = sampleRate;
    }

    public String globalInfo() {
        return null;
    }

    public Instances determineOutputFormat(Instances inputFormat) {
        Instances instances = null;
        FastVector<Attribute> attributes = new FastVector<>(melFreqCepsCo.length);
        for (int i = 0; i < (melFreqCepsCo.length); i++) {
            attributes.addElement(new Attribute("MFCC_att" + interval + String.valueOf(i + 1)));
        }

        FastVector<String> classValues = new FastVector<>(inputFormat.classAttribute().numValues());
        for (int i = 0; i < inputFormat.classAttribute().numValues(); i++)
            classValues.addElement(inputFormat.classAttribute().value(i));
        attributes.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), classValues));

        instances = new Instances("", attributes, 0);

        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }

    public Instances determineOutputFormatForFirstChannel(Instances instances) {
        try {
            return determineOutputFormat(instances);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        // TODO: not sure on how to oconvert this. need to have a think.
        throw new NotImplementedException("This method is not implemented for single transformations.");
    }

    @Override
    public Instance transform(Instance inst) {
        // TODO: not sure on how to oconvert this. need to have a think.
        throw new NotImplementedException("This method is not implemented for single transformations.");
    }

    /**
     *
     * @param instances Univariate instances.
     * @return Relational instances containing MFCC information.
     * @throws Exception
     */
    public Instances transform(Instances instances) {
        Instances[] MFCCs = new Instances[instances.size()];
        Instances flatMFCCs = null;
        Instances MFCCInstances = null;
        double[][] spectrogram = null;
        double cumalativeFilteredVals = 0;
        double[] signal = new double[instances.get(0).numAttributes() - 1];

        // Check whether samplerate info is in file, deletes before moving on.
        if ((instances.attribute("samplerate") != null) && checkForSampleRate) {
            sampleRate = (int) instances.get(0).value(instances.attribute("samplerate"));
            instances.deleteAttributeAt(instances.attribute("samplerate").index());
        }
        if (sampleRate == 0) {
            sampleRate = nfft;
        } else {
            windowLength = (int) ((windowDuration / 1000.0) * (double) sampleRate);
            overlapLength = (int) ((overlapDuration / 1000.0) * (double) sampleRate);
        }

        this.spectrogram.setWindowLength(windowLength);
        this.spectrogram.setOverlap(overlapLength);

        if (windowLength > nfft) {
            System.out.print("NOTE: NFFT < window length, increased from " + nfft);
            nfft = nearestPowerOF2(windowLength);
            System.out.println(" to " + nfft);
        }

        for (int i = 0; i < instances.size(); i++) {
            MFCCInstances = null;
            spectrogram = null;
            cumalativeFilteredVals = 0;
            signal = new double[instances.get(0).numAttributes() - 1];

            for (int j = 0; j < instances.get(i).numAttributes() - 1; j++) {
                signal[j] = instances.get(i).value(j);
            }

            spectrogram = this.spectrogram.spectrogram(signal, windowLength, overlapLength, nfft);

            // Performed to create Periodogram estimate of the power spectrum.
            for (int j = 0; j < spectrogram.length; j++) {
                for (int k = 0; k < spectrogram[j].length; k++) {
                    spectrogram[j][k] = (1 / (double) spectrogram[j].length) * Math.pow(spectrogram[j][k], 2);
                }
            }

            filterBank = createFilterBanks();
            melFreqCepsCo = new double[spectrogram.length][filterBank.length];

            for (int j = 0; j < spectrogram.length; j++) {
                for (int k = 0; k < filterBank.length; k++) {
                    cumalativeFilteredVals = 0;
                    for (int l = 0; l < spectrogram[j].length; l++) {
                        cumalativeFilteredVals += (spectrogram[j][l] * filterBank[k][l]);
                    }
                    melFreqCepsCo[j][k] = cumalativeFilteredVals == 0 ? 0 : Math.log(cumalativeFilteredVals);
                }
            }

            for (int j = 0; j < melFreqCepsCo.length; j++) {
                melFreqCepsCo[j] = dct.transform(melFreqCepsCo[j], TransformType.FORWARD);
            }

            MFCCInstances = determineOutputFormat(instances.get(i).dataset());
            double[] temp;
            for (int j = 0; j < Math.floor(numFilterBanks / 2); j++) {
                temp = new double[melFreqCepsCo.length + 1];
                for (int k = 0; k < melFreqCepsCo.length; k++) {
                    temp[k] = (-1) * melFreqCepsCo[k][j];
                }
                temp[temp.length - 1] = instances.get(i).value(instances.get(i).numAttributes() - 1);
                MFCCInstances.add(new DenseInstance(1.0, temp));
            }

            MFCCs[i] = MFCCInstances;
        }

        // Rearrange data
        Instances[] temp = new Instances[MFCCs[0].size()];
        for (int i = 0; i < temp.length; i++) {
            temp[i] = new Instances(MFCCs[0], 0);
            for (int j = 0; j < MFCCs.length; j++) {
                temp[i].add(MFCCs[j].get(i));
            }
        }

        return MultivariateInstanceTools.concatinateInstances(temp);
    }

    private double[][] createFilterBanks() {
        filterBank = new double[numFilterBanks][nfft / 2];
        double[] filterPeaks = new double[numFilterBanks + 2];
        // Local overload for holding Mel conversion.
        double lowerFreq = 1125 * Math.log(1 + (this.lowerFreq / (double) 700));
        double upperFreq = 1125 * Math.log(1 + (this.upperFreq / (double) 700));
        double step = (upperFreq - lowerFreq) / (filterPeaks.length - 1);

        for (int i = 0; i < filterPeaks.length; i++) {
            filterPeaks[i] = lowerFreq + (step * i);
        }

        // Back to hertz.
        for (int i = 0; i < filterPeaks.length; i++) {
            filterPeaks[i] = 700 * (Math.exp(filterPeaks[i] / 1125) - 1);
        }

        for (int i = 0; i < filterPeaks.length; i++) {
            filterPeaks[i] = Math.floor((nfft + 1) * (filterPeaks[i] / this.sampleRate));
        }

        // Create Filter Banks.
        for (int i = 0; i < filterBank.length; i++) {
            for (int j = 0; j < filterBank[i].length; j++) {
                if (j >= filterPeaks[i] && j <= filterPeaks[i + 1]) {
                    filterBank[i][j] = ((j - filterPeaks[i]) / (filterPeaks[i + 1] - filterPeaks[i]));
                }
                if (j > filterPeaks[i + 1] && j < filterPeaks[i + 2]) {
                    filterBank[i][j] = ((filterPeaks[i + 2] - j) / (filterPeaks[i + 2] - filterPeaks[i + 1]));
                }
                if (j > filterPeaks[i + 2]) {
                    filterBank[i][j] = 0;
                }
                if (j < filterPeaks[i] || (j == 0 && filterPeaks[i] == 0)) {
                    filterBank[i][j] = 0;
                }
                if (Double.isNaN(filterBank[i][j])) {
                    filterBank[i][j] = 0;
                }
            }
        }

        return filterBank;
    }

    private int nearestPowerOF2(int x) {
        float power = (float) (Math.log(x) / Math.log(2));
        int m = (int) Math.ceil(power);
        nfft = (int) Math.pow(2.0, (double) m);
        return nfft;
    }

    public static void main(String[] args) {
        MFCC mfcc = new MFCC();
        Instances[] data = new Instances[2];
        data[0] = loadDataNullable("D:\\Test\\Datasets\\Truncated\\HeartbeatSound\\Heartbeatsound");
        mfcc.setSampleRate(16000);
        System.out.println(data[0].get(0).toString());
        try {
            data[1] = mfcc.transform(data[0]);
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println();
        System.out.println(data[1].get(0));
        Instance[] temp = MultivariateInstanceTools.splitMultivariateInstanceWithClassVal(data[1].get(0));
        System.out.println(temp[0]);
    }




}
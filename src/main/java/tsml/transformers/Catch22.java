/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package tsml.transformers;

import utilities.GenericTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;

import static utilities.ArrayUtilities.mean;
import static utilities.ClusteringUtilities.zNormalise;
import static utilities.GenericTools.*;
import static utilities.StatisticalUtilities.median;
import static utilities.StatisticalUtilities.standardDeviation;
import static utilities.Utilities.extractTimeSeries;

/**
 * Transform for the catch22 set of features.
 *
 * C.H. Lubba, S.S. Sethi, P. Knaute, S.R. Schultz, B.D. Fulcher, N.S. Jones.
 * catch22: CAnonical Time-series CHaracteristics. Data Mining and Knowledge
 * Discovery (2019)
 *
 * Implementation based on C and Matlab code provided on authors github:
 * https://github.com/chlubba/catch22
 *
 * @author Matthew Middlehurst
 */
public class Catch22 implements Transformer {

    // z-norm before transform
    private boolean norm = false;
    // specifically normalise for the outlier stats, which can take a long time with
    // large positive/negative values
    private boolean outlierNorm = false;

    // for summary stat by index
    private int currentSeriesIndex = Integer.MIN_VALUE;
    private double idxMin;
    private double idxMax;
    private double idxMean;
    private FFT.Complex[] idxFFT;
    private double[] idxAC;
    private double[] idxOutlierSeries;
    private double[] idxSeries;

    public Catch22() {
    }

    public void setNormalise(boolean b) {
        this.norm = b;
    }

    public void setOutlierNormalise(boolean b) {
        this.outlierNorm = b;
    }

    @Override
    public Instance transform(Instance inst) {
        double[] arr = extractTimeSeries(inst);
        double cls = inst.classIndex() >= 0 ? inst.classValue() : Double.MIN_VALUE;
        double[] featureSet = transform(arr, cls);
        return new DenseInstance(1, featureSet);
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        //could do this across all dimensions.
        double[][] out = new double[inst.getNumDimensions()][];
        int i = 0;
        for(TimeSeries ts : inst){
            out[i++] = transform(ts.toValueArray());
        }
        
        //create a new output instance with the ACF data.
        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 1; i <= 22; i++) {
            atts.add(new Attribute("att" + i));
        }
        if (data.classIndex() >= 0) atts.add(data.classAttribute());
        Instances transformedData = new Instances("Catch22Transform", atts, data.numInstances());
        if (data.classIndex() >= 0) transformedData.setClassIndex(transformedData.numAttributes()-1);
        return transformedData;
    }

    public double[] transform(double[] series){
        return transform(series, Double.MIN_VALUE);
    }

    //no class value in series
    public double[] transform(double[] series, double classValue){
        int atts = classValue == Double.MIN_VALUE ? 22 : 23;
        double[] featureSet = new double[atts];

        double[] arr;
        double[] outlierArr;
        if (norm) {
            arr = new double[series.length];
            System.arraycopy(series, 0, arr, 0, series.length);
            zNormalise(arr);
            outlierArr = arr;
        } else if (outlierNorm) {
            arr = series;
            outlierArr = new double[series.length];
            System.arraycopy(series, 0, outlierArr, 0, series.length);
            zNormalise(outlierArr);
        } else {
            arr = series;
            outlierArr = series;
        }

        // can reduce amount of computation by pre-computing stats and transforms
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] < min) {
                min = arr[i];
            }
            if (arr[i] > max) {
                max = arr[i];
            }
        }

        double mean = mean(arr);

        int length = (int) FFT.MathsPower2.roundPow2((float) arr.length);
        if (length < arr.length)
            length *= 2;
        FFT.Complex[] fft = new FFT.Complex[length];
        for (int i = 0; i < arr.length; i++) {
            fft[i] = new FFT.Complex(arr[i] - mean, 0);
        }
        for (int i = arr.length; i < length; i++) {
            fft[i] = new FFT.Complex(0, 0);
        }
        FFT t = new FFT();
        t.fft(fft, length);

        FFT.Complex[] fftClone = new FFT.Complex[length];
        for (int i = 0; i < length; i++) {
            fftClone[i] = (FFT.Complex) fft[i].clone();
        }
        double[] ac = autoCorr(arr, fftClone);

        featureSet[0] = histMode5DN(arr, min, max);
        featureSet[1] = histMode10DN(arr, min, max);
        featureSet[2] = binaryStatsMeanLongstretch1SB(arr, mean);
        featureSet[3] = outlierIncludeP001mdrmdDN(outlierArr);
        featureSet[4] = outlierIncludeN001mdrmdDN(outlierArr);
        featureSet[5] = f1ecacCO(ac);
        featureSet[6] = firstMinacCO(ac);
        featureSet[7] = summariesWelchRectArea51SP(arr, fft);
        featureSet[8] = summariesWelchRectCentroidSP(arr, fft);
        featureSet[9] = localSimpleMean3StderrFC(arr);
        featureSet[10] = trev1NumCO(arr);
        featureSet[11] = histogramAMIeven25CO(arr, min, max);
        featureSet[12] = autoMutualInfoStats40GaussianFmmiIN(ac);
        featureSet[13] = hrvClassicPnn40MD(arr);
        featureSet[14] = binaryStatsDiffLongstretch0SB(arr);
        featureSet[15] = motifThreeQuantileHhSB(arr);
        featureSet[16] = localSimpleMean1TauresratFC(arr, ac);
        featureSet[17] = embed2DistTauDExpfitMeandiffCO(arr, ac);
        featureSet[18] = fluctAnal2Dfa5012LogiPropR1SC(arr);
        featureSet[19] = fluctAnal2Rsrangefit501LogiPropR1SC(arr);
        featureSet[20] = transitionMatrix3acSumdiagcovSB(arr, ac);
        featureSet[21] = periodicityWangTh001PD(arr);

        if (classValue > Double.MIN_VALUE) featureSet[22] = classValue;

        for (int i = 0; i < featureSet.length; i++){
            if (Double.isNaN(featureSet[i]) || Double.isInfinite(featureSet[i])){
                featureSet[i] = 0;
            }
        }

        return featureSet;
    }

    public double getSummaryStatByIndex(int summaryStatIndex, int seriesIndex, double[] series) throws Exception {
        if (seriesIndex != currentSeriesIndex) {
            currentSeriesIndex = seriesIndex;
            idxMin = Double.MAX_VALUE;
            idxMax = Double.MIN_VALUE;
            idxMean = Double.MIN_VALUE;
            idxFFT = null;
            idxAC = null;
            idxOutlierSeries = null;

            if (norm) {
                idxSeries = new double[series.length];
                System.arraycopy(series, 0, idxSeries, 0, series.length);
                zNormalise(idxSeries);
            } else {
                idxSeries = series;
            }
        }

        switch (summaryStatIndex) {
            case 0:
            case 1:
            case 11:
                if (idxMin == Double.MAX_VALUE) {
                    for (int i = 0; i < idxSeries.length; i++) {
                        if (idxSeries[i] < idxMin) {
                            idxMin = idxSeries[i];
                        }
                        if (idxSeries[i] > idxMax) {
                            idxMax = idxSeries[i];
                        }
                    }
                }
                break;
            case 2:
                if (idxMean == Double.MIN_VALUE) {
                    idxMean = mean(idxSeries);
                }
                break;
            case 3:
            case 4:
                if (idxOutlierSeries == null) {
                    if (outlierNorm && !norm) {
                        idxOutlierSeries = new double[idxSeries.length];
                        for (int i = 0; i < idxSeries.length; i++) {
                            idxOutlierSeries[i] = idxSeries[i];
                        }
                        zNormalise(idxOutlierSeries);
                    } else {
                        idxOutlierSeries = idxSeries;
                    }
                }
                break;
            case 7:
            case 8:
                if (idxFFT == null) {
                    if (idxMean == Double.MIN_VALUE) {
                        idxMean = mean(idxSeries);
                    }

                    int length = (int) FFT.MathsPower2.roundPow2((float) idxSeries.length);
                    if (length < idxSeries.length)
                        length *= 2;
                    idxFFT = new FFT.Complex[length];
                    for (int i = 0; i < idxSeries.length; i++) {
                        idxFFT[i] = new FFT.Complex(idxSeries[i] - idxMean, 0);
                    }
                    for (int i = idxSeries.length; i < length; i++) {
                        idxFFT[i] = new FFT.Complex(0, 0);
                    }
                    FFT t = new FFT();
                    t.fft(idxFFT, length);
                }
                break;
            case 5:
            case 6:
            case 12:
            case 16:
            case 17:
            case 20:
                if (idxAC == null) {
                    if (idxFFT == null) {
                        if (idxMean == Double.MIN_VALUE) {
                            idxMean = mean(idxSeries);
                        }

                        int length = (int) FFT.MathsPower2.roundPow2((float) idxSeries.length);
                        if (length < idxSeries.length)
                            length *= 2;
                        idxFFT = new FFT.Complex[length];
                        for (int i = 0; i < idxSeries.length; i++) {
                            idxFFT[i] = new FFT.Complex(idxSeries[i] - idxMean, 0);
                        }
                        for (int i = idxSeries.length; i < length; i++) {
                            idxFFT[i] = new FFT.Complex(0, 0);
                        }
                        FFT t = new FFT();
                        t.fft(idxFFT, length);
                    }

                    FFT.Complex[] fftClone = new FFT.Complex[idxFFT.length];
                    for (int i = 0; i < idxFFT.length; i++) {
                        fftClone[i] = (FFT.Complex) idxFFT[i].clone();
                    }
                    idxAC = autoCorr(idxSeries, fftClone);
                }
                break;
        }

        switch (summaryStatIndex) {
            case 0:
                return histMode5DN(idxSeries, idxMin, idxMax);
            case 1:
                return histMode10DN(idxSeries, idxMin, idxMax);
            case 2:
                return binaryStatsMeanLongstretch1SB(idxSeries, idxMean);
            case 3:
                return outlierIncludeP001mdrmdDN(idxOutlierSeries);
            case 4:
                return outlierIncludeN001mdrmdDN(idxOutlierSeries);
            case 5:
                return f1ecacCO(idxAC);
            case 6:
                return firstMinacCO(idxAC);
            case 7:
                return summariesWelchRectArea51SP(idxSeries, idxFFT);
            case 8:
                return summariesWelchRectCentroidSP(idxSeries, idxFFT);
            case 9:
                return localSimpleMean3StderrFC(idxSeries);
            case 10:
                return trev1NumCO(idxSeries);
            case 11:
                return histogramAMIeven25CO(idxSeries, idxMin, idxMax);
            case 12:
                return autoMutualInfoStats40GaussianFmmiIN(idxAC);
            case 13:
                return hrvClassicPnn40MD(idxSeries);
            case 14:
                return binaryStatsDiffLongstretch0SB(idxSeries);
            case 15:
                return motifThreeQuantileHhSB(idxSeries);
            case 16:
                return localSimpleMean1TauresratFC(idxSeries, idxAC);
            case 17:
                return embed2DistTauDExpfitMeandiffCO(idxSeries, idxAC);
            case 18:
                return fluctAnal2Dfa5012LogiPropR1SC(idxSeries);
            case 19:
                return fluctAnal2Rsrangefit501LogiPropR1SC(idxSeries);
            case 20:
                return transitionMatrix3acSumdiagcovSB(idxSeries, idxAC);
            case 21:
                return periodicityWangTh001PD(idxSeries);
            default:
                throw new Exception("Invalid Catch22 summary stat index.");
        }
    }

    public static String getSummaryStatNameByIndex(int summaryStatIndex) throws Exception {
        switch(summaryStatIndex){
            case 0: return "DN_HistogramMode_5";
            case 1: return "DN_HistogramMode_10";
            case 2: return "SB_BinaryStats_mean_longstretch1";
            case 3: return "DN_OutlierInclude_p_001_mdrmd";
            case 4: return "DN_OutlierInclude_n_001_mdrmd";
            case 5: return "CO_f1ecac";
            case 6: return "CO_FirstMin_ac";
            case 7: return "SP_Summaries_welch_rect_area_5_1";
            case 8: return "SP_Summaries_welch_rect_centroid";
            case 9: return "FC_LocalSimple_mean3_stderr";
            case 10: return "CO_trev_1_num";
            case 11: return "CO_HistogramAMI_even_2_5";
            case 12: return "IN_AutoMutualInfoStats_40_gaussian_fmmi";
            case 13: return "MD_hrv_classic_pnn40";
            case 14: return "SB_BinaryStats_diff_longstretch0";
            case 15: return "SB_MotifThree_quantile_hh";
            case 16: return "FC_LocalSimple_mean1_tauresrat";
            case 17: return "CO_Embed2_Dist_tau_d_expfit_meandiff";
            case 18: return "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1";
            case 19: return "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1";
            case 20: return "SB_TransitionMatrix_3ac_sumdiagcov";
            case 21: return "PD_PeriodicityWang_th0_01";
            default: throw new Exception("Invalid Catch22 summary stat index.");
        }
    }

    // Mode of z-scored distribution (5-bin histogram)
    private static double histMode5DN(double[] arr, double min, double max) {
        return histogramMode(arr, 5, min, max);
    }

    // Mode of z-scored distribution (10-bin histogram)
    private static double histMode10DN(double[] arr, double min, double max) {
        return histogramMode(arr, 10, min, max);
    }

    // Longest period of consecutive values above the mean
    private static double binaryStatsMeanLongstretch1SB(double[] arr, double mean) {
        int[] meanBinary = new int[arr.length];
        // double mean = mean(arr);
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] - mean > 0) {
                meanBinary[i] = 1;
            }
        }

        return longStretch(meanBinary, 1);
    }

    // Time intervals between successive extreme events above the mean
    private static double outlierIncludeP001mdrmdDN(double[] arr) {
        return outlierInclude(arr);
    }

    // Time intervals between successive extreme events below the mean
    private static double outlierIncludeN001mdrmdDN(double[] arr) {
        double[] newArr = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            newArr[i] = -arr[i];
        }

        return outlierInclude(newArr);
    }

    // First 1/e crossing of autocorrelation function
    private static double f1ecacCO(double[] ac) {
        double threshold = 0.36787944117144233; // 1/Math.exp(1);
        // double[] ac = autoCorr(arr);

        for (int i = 1; i < ac.length; i++) {
            if ((ac[i - 1] - threshold) * (ac[i] - threshold) < 0) {
                return i;
            }
        }

        return ac.length;
    }

    // First minimum of autocorrelation function
    private static double firstMinacCO(double[] ac) {
        // double[] ac = autoCorr(arr);

        for (int i = 1; i < ac.length - 1; i++) {
            if (ac[i] < ac[i - 1] && ac[i] < ac[i + 1]) {
                return i;
            }
        }

        return ac.length;
    }

    // Total power in lowest fifth of frequencies in the Fourier power spectrum
    private static double summariesWelchRectArea51SP(double[] arr, FFT.Complex[] fft) {
        return summariesWelchRect(arr, false, fft);
    }

    // Centroid of the Fourier power spectrum
    private static double summariesWelchRectCentroidSP(double[] arr, FFT.Complex[] fft) {
        return summariesWelchRect(arr, true, fft);
    }

    // Mean error from a rolling 3-sample mean forecasting
    private static double localSimpleMean3StderrFC(double[] arr) {
        if (arr.length - 3 < 3)
            return 0;
        double[] res = localSimpleMean(arr, 3);

        return standardDeviation(res, false);
    }

    // Time-reversibility statistic, ((x_t+1 − x_t)^3)_t
    private static double trev1NumCO(double[] arr) {
        double[] y = new double[arr.length - 1];

        for (int i = 0; i < y.length; i++) {
            y[i] = Math.pow(arr[i + 1] - arr[i], 3);
        }

        return mean(y);
    }

    // Automutual information, m = 2, τ = 5
    private static double histogramAMIeven25CO(double[] arr, double min, double max) {
        // double min = min(arr)-0.1;
        // double max = max(arr)+0.1;
        double newMin = min - 0.1;
        double newMax = max + 0.1;
        double binWidth = (newMax - newMin) / 5;

        double[][] histogram = new double[5][5];
        double[] sumx = new double[5];
        double[] sumy = new double[5];
        double v = 1.0 / (arr.length - 2);
        for (int i = 0; i < arr.length - 2; i++) {
            int idx1 = (int) ((arr[i] - newMin) / binWidth);
            int idx2 = (int) ((arr[i + 2] - newMin) / binWidth);

            histogram[idx1][idx2] += v;
            sumx[idx1] += v;
            sumy[idx2] += v;
        }

        double sum = 0;
        for (int i = 0; i < 5; i++) {
            for (int n = 0; n < 5; n++) {
                if (histogram[i][n] > 0) {
                    sum += histogram[i][n] * Math.log(histogram[i][n] / sumx[i] / sumy[n]);
                }
            }
        }

        return sum;
    }

    // First minimum of the automutual information function
    private static double autoMutualInfoStats40GaussianFmmiIN(double[] ac) {
        int tau = Math.min(40, (int) Math.ceil(ac.length / 2));

        double[] diffs = new double[tau - 1];
        double prev = -0.5 * Math.log(1 - Math.pow(ac[1], 2));
        for (int i = 0; i < diffs.length; i++) {
            double corr = -0.5 * Math.log(1 - Math.pow(ac[i + 2], 2));
            diffs[i] = corr - prev;
            prev = corr;
        }

        for (int i = 0; i < diffs.length - 1; i++) {
            if (diffs[i] * diffs[i + 1] < 0 && diffs[i] < 0) {
                return i + 1;
            }
        }

        return tau;
    }

    // Proportion of successive differences exceeding 0.04σ (Mietus 2002)
    private static double hrvClassicPnn40MD(double[] arr) {
        double[] diffs = new double[arr.length - 1];
        for (int i = 0; i < diffs.length; i++) {
            diffs[i] = Math.abs(arr[i + 1] - arr[i]) * 1000;
        }

        double sum = 0;
        for (int i = 0; i < diffs.length; i++) {
            if (diffs[i] > 40) {
                sum++;
            }
        }

        return sum / diffs.length;
    }

    // Longest period of successive incremental decreases
    private static double binaryStatsDiffLongstretch0SB(double[] arr) {
        int[] diffBinary = new int[arr.length - 1];
        for (int i = 0; i < diffBinary.length; i++) {
            if (arr[i + 1] - arr[i] >= 0) {
                diffBinary[i] = 1;
            }
        }

        return longStretch(diffBinary, 0);
    }

    // Shannon entropy of two successive letters in equiprobable 3-letter
    // symbolization
    private static double motifThreeQuantileHhSB(double[] arr) {
        GenericTools.SortIndexAscending sort = new GenericTools.SortIndexAscending(arr);
        Integer[] indicies = sort.getIndicies();
        Arrays.sort(indicies, sort);

        ArrayList<ArrayList<Integer>> p = new ArrayList<>();
        int[] bins = new int[arr.length];
        double q1 = arr.length / 3;
        double q2 = q1 * 2;
        p.add(new ArrayList<>());
        for (int i = 0; i <= q1; i++) {
            bins[indicies[i]] = 0;
            p.get(0).add(indicies[i]);
        }
        p.add(new ArrayList<>());
        for (int i = (int) Math.ceil(q1 + 0.1); i <= q2; i++) {
            bins[indicies[i]] = 1;
            p.get(1).add(indicies[i]);
        }
        p.add(new ArrayList<>());
        for (int i = (int) Math.ceil(q2 + 0.1); i < indicies.length; i++) {
            bins[indicies[i]] = 2;
            p.get(2).add(indicies[i]);
        }

        double sum = 0;

        for (int i = 0; i < 3; i++) {
            ArrayList<Integer> o = p.get(i);
            o.remove((Integer) (arr.length - 1));

            for (int n = 0; n < 3; n++) {
                double sum2 = 0;

                for (int j = 0; j < o.size(); j++) {
                    if (bins[o.get(j) + 1] == n) {
                        sum2++;
                    }
                }

                if (sum2 > 0) {
                    sum2 /= (arr.length - 1);
                    sum += sum2 * Math.log(sum2);
                }
            }
        }

        return -sum;
    }

    // Change in correlation length after iterative differencing
    private static double localSimpleMean1TauresratFC(double[] arr, double[] ac) {
        if (arr.length - 1 < 1)
            return 0;
        double[] res = localSimpleMean(arr, 1);

        int length = (int) FFT.MathsPower2.roundPow2((float) res.length);
        if (length < res.length)
            length *= 2;

        FFT.Complex[] fft = new FFT.Complex[length];
        double mean = mean(res);
        for (int i = 0; i < res.length; i++) {
            fft[i] = new FFT.Complex(res[i] - mean, 0);
        }
        for (int i = res.length; i < length; i++) {
            fft[i] = new FFT.Complex(0, 0);
        }

        FFT t = new FFT();
        t.fft(fft, length);

        double[] resAc = autoCorr(res, fft);

        return (double) acFirstZero(resAc) / acFirstZero(ac);
    }

    // Exponential fit to successive distances in 2-d embedding space
    private static double embed2DistTauDExpfitMeandiffCO(double[] arr, double[] ac) {
        int tau = acFirstZero(ac);
        if (tau > arr.length / 10) {
            tau = arr.length / 10;
        }

        double[] d = new double[arr.length - tau - 1];
        double dMean = 0;
        for (int i = 0; i < d.length; i++) {
            double n = Math.sqrt(Math.pow(arr[i + 1] - arr[i], 2) + Math.pow(arr[i + tau + 1] - arr[i + tau], 2));
            d[i] = n;
            dMean += n;
        }
        dMean /= arr.length - tau - 1;

        double min = min(d);
        double max = max(d);
        double range = max - min;
        int numBins = (int) Math.ceil(range / (3.5 * standardDeviation(d, false, dMean) /
                Math.pow(d.length, 0.3333333333333333)));
        double binWidth = range / numBins;

        if (numBins == 0)
            return Double.NaN; // check this out

        double[] histogram = new double[numBins];
        for (double val : d) {
            int idx = (int) ((val - min) / binWidth);
            if (idx >= numBins)
                idx = numBins - 1;
            histogram[idx]++;
        }

        double sum = 0;
        for (int i = 0; i < numBins; i++) {
            double center = ((min + binWidth * (i)) * 2 + binWidth) / 2;
            double n = Math.exp((-center) / dMean) / dMean;
            if (n < 0)
                n = 0;

            sum += Math.abs(histogram[i] / d.length - n);
        }

        return sum / numBins;
    }

    // Proportion of slower timescale fluctuations that scale with DFA (50%
    // sampling)
    private static double fluctAnal2Dfa5012LogiPropR1SC(double[] arr) {
        double[] cs = new double[arr.length / 2];
        cs[0] = arr[0];
        for (int i = 1; i < cs.length; i++) {
            cs[i] = cs[i - 1] + arr[i * 2];
        }

        return fluctProp(cs, arr.length, true);
    }

    // Proportion of slower timescale fluctuations that scale with linearly rescaled
    // range fits
    private static double fluctAnal2Rsrangefit501LogiPropR1SC(double[] arr) {
        double[] cs = new double[arr.length];
        cs[0] = arr[0];
        for (int i = 1; i < arr.length; i++) {
            cs[i] = cs[i - 1] + arr[i];
        }

        return fluctProp(cs, arr.length, false);
    }

    // Trace of covariance of transition matrix between symbols in 3-letter alphabet
    private static double transitionMatrix3acSumdiagcovSB(double[] arr, double[] ac) {
        // int numGroups = 3;
        int tau = acFirstZero(ac);
        int dsSize = (arr.length - 1) / tau + 1;
        double[] ds = new double[dsSize];
        for (int i = 0; i < dsSize; i++) {
            ds[i] = arr[i * tau];
        }

        GenericTools.SortIndexAscending sort = new GenericTools.SortIndexAscending(ds);
        Integer[] indicies = sort.getIndicies();
        Arrays.sort(indicies, sort);

        int[] bins = new int[ds.length];
        double q1 = ds.length / 3;
        double q2 = q1 * 2;
        for (int i = 0; i <= q1; i++) {
            bins[indicies[i]] = 0;
        }
        for (int i = (int) Math.ceil(q1 + 0.1); i <= q2; i++) {
            bins[indicies[i]] = 1;
        }
        for (int i = (int) Math.ceil(q2 + 0.1); i < indicies.length; i++) {
            bins[indicies[i]] = 2;
        }

        double[][] t = new double[3][3];
        for (int i = 0; i < dsSize - 1; i++) {
            t[bins[i + 1]][bins[i]] += 1;
        }

        for (int i = 0; i < 3; i++) {
            for (int n = 0; n < 3; n++) {
                t[i][n] /= (dsSize - 1);
            }
        }

        double[] means = new double[3];
        for (int i = 0; i < 3; i++) {
            means[i] = mean(t[i]);
        }

        double[][] cov = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int n = i; n < 3; n++) {
                double covariance = 0;
                for (int j = 0; j < 3; j++) {
                    covariance += (t[i][j] - means[i]) * (t[n][j] - means[n]);
                }
                covariance /= 2;

                cov[i][n] = covariance;
                cov[n][i] = covariance;
            }
        }

        double sum = 0;
        for (int i = 0; i < 3; i++) {
            sum += cov[i][i];
        }

        return sum;
    }

    // Periodicity measure of (Wang et al. 2007)
    private static double periodicityWangTh001PD(double[] arr) {
        double[] ySpline = splineFit(arr);

        double[] ySub = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            ySub[i] = arr[i] - ySpline[i];
        }

        int acmax = (int) Math.ceil(arr.length / 3.0);
        double[] acf = new double[acmax];
        for (int tau = 1; tau <= acmax; tau++) {
            double covariance = 0;
            for (int i = 0; i < arr.length - tau; i++) {
                covariance += ySub[i] * ySub[i + tau];

            }
            acf[tau - 1] = covariance / (arr.length - tau);
        }

        int[] troughs = new int[acmax];
        int[] peaks = new int[acmax];
        int nTroughs = 0;
        int nPeaks = 0;
        for (int i = 1; i < acmax - 1; i++) {
            double slopeIn = acf[i] - acf[i - 1];
            double slopeOut = acf[i + 1] - acf[i];

            if (slopeIn < 0 & slopeOut > 0) {
                troughs[nTroughs] = i;
                nTroughs += 1;
            } else if (slopeIn > 0 & slopeOut < 0) {
                peaks[nPeaks] = i;
                nPeaks += 1;
            }
        }

        int out = 0;
        for (int i = 0; i < nPeaks; i++) {
            int iPeak = peaks[i];
            double thePeak = acf[iPeak];

            int j = -1;
            while (troughs[j + 1] < iPeak && j + 1 < nTroughs) {
                j++;
            }
            if (j == -1) {
                continue;
            }

            int iTrough = troughs[j];
            double theTrough = acf[iTrough];

            if (thePeak - theTrough < 0.01)
                continue;

            if (thePeak < 0)
                continue;

            out = iPeak;
            break;
        }

        return out;
    }

    private static double histogramMode(double[] arr, int numBins, double min, double max) {
        // double min = Double.MAX_VALUE;
        // double max = Double.MIN_VALUE;
        // for(int i = 0; i < arr.length; i++) {
        // if (arr[i] < min) {
        // min = arr[i];
        // }
        // else if (arr[i] > max) {
        // max = arr[i];
        // }
        // }

        double binWidth = (max - min) / numBins;
        double[] histogram = new double[numBins];
        for (double val : arr) {
            int idx = (int) ((val - min) / binWidth);
            if (idx >= numBins)
                idx = numBins - 1;
            histogram[idx]++;
        }

        double[] edges = new double[numBins + 1];
        for (int i = 0; i < edges.length; i++) {
            edges[i] = i * binWidth + min;
        }

        double maxCount = 0;
        int numMaxs = 1;
        double maxSum = 0;
        for (int i = 0; i < numBins; i++) {
            if (histogram[i] > maxCount) {
                maxCount = histogram[i];
                numMaxs = 1;
                maxSum = (edges[i] + edges[i + 1]) * 0.5;
            } else if (histogram[i] == maxCount) {
                numMaxs += 1;
                maxSum += (edges[i] + edges[i + 1]) * 0.5;
            }
        }

        return maxSum / numMaxs;
    }

    private static double longStretch(int[] binary, int val) {
        double lastVal = 0;
        double maxStretch = 0;
        for (int i = 0; i < binary.length; i++) {
            if (binary[i] != val || i == binary.length - 1) {
                double stretch = i - lastVal;
                if (stretch > maxStretch) {
                    maxStretch = stretch;
                }
                lastVal = i;
            }
        }

        return maxStretch;
    }

    private static double outlierInclude(double[] arr) {
        double total = 0;
        double threshold = 0;

        for (double v : arr) {
            if (v >= 0) {
                total++;
                if (v > threshold) {
                    threshold = v;
                }
            }
        }

        if (threshold < 0.01)
            return 0;

        int numThresholds = (int) (threshold / 0.01) + 1;
        double[] means = new double[numThresholds];
        double[] dists = new double[numThresholds];
        double[] medians = new double[numThresholds];
        for (int i = 0; i < numThresholds; i++) {
            double d = i * 0.01;

            ArrayList<Double> r = new ArrayList<>(arr.length);
            for (int n = 0; n < arr.length; n++) {
                if (arr[n] >= d) {
                    r.add(n + 1.0);
                }
            }

            if (r.size() == 0)
                continue;

            double[] diff = new double[r.size() - 1];
            for (int n = 0; n < diff.length; n++) {
                diff[n] = r.get(n + 1) - r.get(n);
            }

            means[i] = mean(diff);
            dists[i] = diff.length * 100.0 / total;

            medians[i] = median(r, false) / (arr.length / 2.0) - 1;
        }

        int mj = 0;
        int fbi = numThresholds - 1;
        for (int i = 0; i < numThresholds; i++) {
            if (dists[i] > 2) {
                mj = i;
            }
            if (Double.isNaN(means[i])) {
                fbi = numThresholds - 1 - i;
            }
        }

        int trimLimit = mj < fbi ? fbi : mj;

        return median(Arrays.copyOf(medians, trimLimit + 1), false);
    }

    private static double[] autoCorr(double[] arr, FFT.Complex[] fft) {
        // int length = (int)FFT.MathsPower2.roundPow2((float)arr.length);
        // if (length < arr.length) length *= 2;
        //
        // FFT.Complex[] fft = new FFT.Complex[length];
        // double mean = mean(arr);
        // for (int i = 0; i < arr.length; i++){
        // fft[i] = new FFT.Complex(arr[i]-mean, 0);
        // }
        // for (int i = arr.length; i < length; i++){
        // fft[i] = new FFT.Complex(0,0);
        // }
        //
        FFT t = new FFT();
        // t.fft(fft, length);

        for (int i = 0; i < fft.length; i++) {
            fft[i].multiply(new FFT.Complex(fft[i].getReal(), -fft[i].getImag()));
        }

        t.inverseFFT(fft, fft.length);

        double[] acf = new double[arr.length];
        float f = fft[0].getReal();
        for (int i = 0; i < arr.length; i++) {
            acf[i] = fft[i].getReal() / f;
        }

        return acf;
    }

    private static double summariesWelchRect(double[] arr, boolean centroid, FFT.Complex[] fft) {
        // int length = (int)FFT.MathsPower2.roundPow2((float)arr.length);
        // if (length < arr.length) length *= 2;
        //
        // FFT.Complex[] fft = new FFT.Complex[length];
        // double mean = mean(arr);
        // for (int i = 0; i < arr.length; i++){
        // fft[i] = new FFT.Complex(arr[i]-mean, 0);
        // }
        // for (int i = arr.length; i < length; i++){
        // fft[i] = new FFT.Complex(0,0);
        // }
        //
        // FFT t = new FFT();
        // t.fft(fft, length);

        int newLength = fft.length / 2 + 1;
        double[] p = new double[newLength];
        p[0] = (Math.pow(fft[0].getMagnitude(), 2) / arr.length) / (2 * Math.PI);
        for (int i = 1; i < newLength - 1; i++) {
            p[i] = ((Math.pow(fft[i].getMagnitude(), 2) / arr.length) * 2) / (2 * Math.PI);
        }
        p[newLength - 1] = (Math.pow(fft[newLength - 1].getMagnitude(), 2) / arr.length) / (2 * Math.PI);

        double[] w = new double[newLength];
        for (int i = 0; i < newLength; i++) {
            w[i] = i * (1.0 / fft.length) * Math.PI * 2;
        }

        if (centroid) {
            double[] cs = new double[newLength];
            cs[0] = p[0];
            for (int i = 1; i < newLength; i++) {
                cs[i] = cs[i - 1] + p[i];
            }

            double threshold = cs[newLength - 1] / 2;
            for (int i = 0; i < newLength; i++) {
                if (cs[i] > threshold) {
                    return w[i];
                }
            }
            return Double.NaN;
        } else {
            double tau = Math.floor(newLength / 5);
            double sum = 0;
            for (int i = 0; i < tau; i++) {
                sum += p[i];
            }

            return sum * (w[1] - w[0]);
        }
    }

    private static double[] localSimpleMean(double[] arr, int trainLength) {
        double[] res = new double[arr.length - trainLength];
        for (int i = 0; i < res.length; i++) {
            double sum = 0;
            for (int n = 0; n < trainLength; n++) {
                sum += arr[i + n];
            }
            res[i] = arr[i + trainLength] - sum / trainLength;
        }
        return res;
    }

    private static int acFirstZero(double[] ac) {
        // double[] ac = autoCorr(arr, fft);

        for (int i = 1; i < ac.length; i++) {
            if (ac[i] < 0) {
                return i;
            }
        }

        return ac.length;
    }

    private static double fluctProp(double[] arr, double ogLength, boolean dfa) {
        // int q = 2;

        ArrayList<Integer> a = new ArrayList<>();
        a.add(5);
        double min = 1.6094379124341003; //Math.log(5);
        double max = Math.log(ogLength/2);
        double inc = (max - min)/49;
        for (int i = 1; i < 50; i++){
            int val = (int)Math.round(Math.exp(min + inc*i));
            if (val != a.get(a.size()-1)){
                a.add(val);
            }
        }
        int nTau = a.size();

        if (a.size() < 12)
            return Double.NaN;

        double[] f = new double[nTau];

        for (int i = 0; i < nTau; i++) {
            int tau = a.get(i);
            int buffSize = arr.length / tau;
            int lag = 0;
            if (buffSize == 0) {
                buffSize = 1;
                lag = 1;
            }

            double[][] buffer = new double[buffSize][tau];
            int count = 0;
            for (int n = 0; n < buffer.length; n++) {
                for (int j = 0; j < tau - lag; j++) {
                    buffer[n][j] = arr[count++];
                }
            }

            double[] d = new double[tau];
            for (int n = 0; n < tau; n++) {
                d[n] = n + 1;
            }

            for (int n = 0; n < buffer.length; n++) {
                double[] co = linearRegression(d, buffer[n], tau, 0);

                for (int j = 0; j < tau; j++) {
                    buffer[n][j] = buffer[n][j] - (co[0] * (j + 1) + co[1]);
                }

                if (dfa) {
                    for (int j = 0; j < tau; j++) {
                        f[i] += buffer[n][j] * buffer[n][j];
                    }
                } else {
                    f[i] += Math.pow(max(buffer[n]) - min(buffer[n]), 2);
                }
            }

            if (dfa) {
                f[i] = Math.sqrt(f[i] / (buffer.length * tau));
            } else {
                f[i] = Math.sqrt(f[i] / buffer.length);
            }
        }

        double[] logA = new double[nTau];
        double[] logF = new double[nTau];
        for (int i = 0; i < nTau; i++) {
            logA[i] = Math.log(a.get(i));
            logF[i] = Math.log(f[i]);
        }

        // int minPoints = 6;
        int nsserr = (nTau - 11); // (nTau - 2*minPoints + 1);
        double[] sserr = new double[nsserr];
        for (int i = 6; i < nTau - 5; i++) { // (nTau - minPoints + 1);
            double[] co = linearRegression(logA, logF, i, 0);
            double[] co2 = linearRegression(logA, logF, nTau - i + 1, i - 1);

            double sum1 = 0;
            for (int n = 0; n < i; n++) {
                sum1 += Math.pow(logA[n] * co[0] + co[1] - logF[n], 2);
            }
            sserr[i - 6] += Math.sqrt(sum1);

            double sum2 = 0;
            for (int n = 0; n < nTau - i + 1; n++) {
                sum2 += Math.pow(logA[n + i - 1] * co2[0] + co2[1] - logF[n + i - 1], 2);
            }
            sserr[i - 6] += Math.sqrt(sum2);
        }

        return (indexOfMin(sserr) + 6) / nTau;
    }

    private static double[] linearRegression(double[] x, double[] y, int n, int lag) {
        double[] co = new double[2];
        double sumx = 0;
        double sumx2 = 0;
        double sumxy = 0;
        double sumy = 0;

        for (int i = lag; i < n + lag; i++) {
            sumx += x[i];
            sumx2 += x[i] * x[i];
            sumxy += x[i] * y[i];
            sumy += y[i];
        }

        double denom = (n * sumx2 - sumx * sumx);
        if (denom != 0) {
            co[0] = (n * sumxy - sumx * sumy) / denom;
            co[1] = (sumy * sumx2 - sumx * sumxy) / denom;
        }

        return co;
    }

    private static double[] splineFit(double[] arr) {
        // int deg = 3;
        // int n = 4;
        // int nBreaks = 3;
        // int peices = 2;
        // int piecesExt = 8;

        int[] breaks = { 0, arr.length / 2 - 1, arr.length - 1 };
        int[] h0 = { breaks[1] - breaks[0], breaks[2] - breaks[1] };
        int[] hCopy = { h0[0], h0[1], h0[0], h0[1] };
        int[] hl = { hCopy[3], hCopy[2], hCopy[1] };

        int[] hlCS = new int[3];
        hlCS[0] = hl[0];
        for (int i = 1; i < 3; i++) {
            hlCS[i] = hlCS[i - 1] + hl[i];
        }

        int[] bl = new int[3];
        for (int i = 0; i < 3; i++) {
            bl[i] = breaks[0] - hlCS[i];
        }

        int[] hr = { hCopy[0], hCopy[1], hCopy[2] };

        int[] hrCS = new int[3];
        hrCS[0] = hr[0];
        for (int i = 1; i < 3; i++) {
            hrCS[i] = hrCS[i - 1] + hr[i];
        }

        int[] br = new int[3];
        for (int i = 0; i < 3; i++) {
            br[i] = breaks[2] + hrCS[i];
        }

        int[] breaksExt = new int[9];
        for (int i = 0; i < 3; i++) {
            breaksExt[i] = bl[2 - i];
            breaksExt[i + 3] = breaks[i];
            breaksExt[i + 6] = br[i];
        }

        int[] hExt = new int[8];
        for (int i = 0; i < 8; i++) {
            hExt[i] = breaksExt[i + 1] - breaksExt[i];
        }

        double[][] coefs = new double[32][4];
        for (int i = 0; i < 32; i += 4) {
            coefs[i][0] = 1;
        }

        int[][] ii = new int[4][8];
        for (int i = 0; i < 8; i++) {
            ii[0][i] = Math.min(i, 7);
            ii[1][i] = Math.min(1 + i, 7);
            ii[2][i] = Math.min(2 + i, 7);
            ii[3][i] = Math.min(3 + i, 7);
        }

        double[] H = new double[32];
        for (int i = 0; i < 32; i++) {
            H[i] = hExt[ii[i % 4][i / 4]];
        }

        for (int k = 1; k < 4; k++) {
            for (int j = 0; j < k; j++) {
                for (int l = 0; l < 32; l++) {
                    coefs[l][j] *= H[l] / (k - j);
                }
            }

            double[][] Q = new double[4][8];
            for (int l = 0; l < 32; l++) {
                for (int m = 0; m < 4; m++) {
                    Q[l % 4][l / 4] += coefs[l][m];
                }
            }

            for (int l = 0; l < 8; l++) {
                for (int m = 1; m < 4; m++) {
                    Q[m][l] += Q[m - 1][l];
                }
            }

            for (int l = 0; l < 32; l++) {
                if (l % 4 > 0) {
                    coefs[l][k] = Q[l % 4 - 1][l / 4];
                }
            }

            double[] fmax = new double[32];
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 4; j++) {
                    fmax[i * 4 + j] = Q[3][i];
                }
            }

            for (int j = 0; j < k + 1; j++) {
                for (int l = 0; l < 32; l++) {
                    coefs[l][j] /= fmax[l];
                }
            }

            for (int i = 0; i < 29; i++) {
                for (int j = 0; j < k + 1; j++) {
                    coefs[i][j] -= coefs[3 + i][j];
                }
            }
            for (int i = 0; i < 32; i += 4) {
                coefs[i][k] = 0;
            }
        }

        double[] scale = new double[32];
        for (int i = 0; i < 32; i++) {
            scale[i] = 1;
        }
        for (int k = 0; k < 3; k++) {
            for (int i = 0; i < (32); i++) {
                scale[i] /= H[i];
            }
            for (int i = 0; i < (32); i++) {
                coefs[i][(3) - (k + 1)] *= scale[i];
            }
        }

        int[][] jj = new int[4][2];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                if (i == 0)
                    jj[i][j] = 4 * (1 + j);
                else
                    jj[i][j] = 3;
            }
        }

        for (int i = 1; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                jj[i][j] += jj[i - 1][j];
            }
        }

        double[][] coefsOut = new double[8][4];
        for (int i = 0; i < 8; i++) {
            int jj_flat = jj[i % 4][i / 4] - 1;
            for (int j = 0; j < 4; j++) {
                coefsOut[i][j] = coefs[jj_flat][j];
            }
        }

        int[] xsB = new int[arr.length * 4];
        int[] indexB = new int[xsB.length];

        int breakInd = 1;
        for (int i = 0; i < arr.length; i++) {
            if (i >= breaks[1] & breakInd < 2)
                breakInd += 1;
            for (int j = 0; j < 4; j++) {
                xsB[i * 4 + j] = i - breaks[breakInd - 1];
                indexB[i * 4 + j] = j + (breakInd - 1) * 4;
            }
        }

        double[] vB = new double[xsB.length];
        for (int i = 0; i < xsB.length; i++) {
            vB[i] = coefsOut[indexB[i]][0];
        }

        for (int i = 1; i < 4; i++) {
            for (int j = 0; j < xsB.length; j++) {
                vB[j] = vB[j] * xsB[j] + coefsOut[indexB[j]][i];
            }
        }

        double[] A = new double[arr.length * 5];
        for (int i = 0; i < A.length; i++) {
            A[i] = 0;
        }
        breakInd = 0;
        for (int i = 0; i < xsB.length; i++) {
            if (i / 4 >= breaks[1])
                breakInd = 1;
            A[(i % 4) + breakInd + (i / 4) * 5] = vB[i];
        }

        double[] x = new double[5];
        // lsqsolve_sub(size, n+1, A, size, y, x);
        // (const int sizeA1, const int sizeA2, const double *A, const int sizeb, const
        // double *b, double *x)

        double[] AT = new double[A.length];
        double[] ATA = new double[25];
        double[] ATb = new double[5];
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < 5; j++) {
                AT[j * arr.length + i] = A[i * 5 + j];
            }
        }

        // matrix_multiply(sizeA2, sizeA1, AT, sizeA1, sizeA2, A, ATA);
        // (const int sizeA1, const int sizeA2, const double *A, const int sizeB1, const
        // int sizeB2, const double *B, double *C){

        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < arr.length; k++) {
                    ATA[i * 5 + j] += AT[i * arr.length + k] * A[k * 5 + j];
                }
            }
        }

        // matrix_times_vector(sizeA2, sizeA1, AT, sizeA1, b, ATb);
        // (const int sizeA1, const int sizeA2, const double *A, const int sizeb, const
        // double *b, double *c)

        for (int i = 0; i < 5; i++) {
            for (int k = 0; k < arr.length; k++) {
                ATb[i] += AT[i * arr.length + k] * arr[k];
            }
        }

        // gauss_elimination(sizeA2, ATA, ATb, x);
        // (int size, double *A, double *b, double *x)

        double[][] AElim = new double[5][5];
        double[] bElim = new double[5];

        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                AElim[i][j] = ATA[i * 5 + j];
            }
            bElim[i] = ATb[i];
        }

        for (int i = 0; i < 5; i++) {
            for (int j = i + 1; j < 5; j++) {
                double factor = AElim[j][i] / AElim[i][i];

                bElim[j] = bElim[j] - factor * bElim[i];

                for (int k = i; k < 5; k++) {
                    AElim[j][k] = AElim[j][k] - factor * AElim[i][k];
                }
            }
        }

        for (int i = 4; i >= 0; i--) {
            double bMinusATemp = bElim[i];
            for (int j = i + 1; j < 5; j++) {
                bMinusATemp -= x[j] * AElim[i][j];
            }

            x[i] = bMinusATemp / AElim[i][i];
        }

        ////

        double[][] C = new double[5][8];
        for (int i = 0; i < 32; i++) {
            int CRow = i % 4 + (i / 4) % 2;
            int CCol = i / 4;

            int coefRow = i % (8);
            int coefCol = i / (8);

            C[CRow][CCol] = coefsOut[coefRow][coefCol];
        }

        double[][] coefsSpline = new double[2][4];
        for (int j = 0; j < 8; j++) {
            int coefCol = j / 2;
            int coefRow = j % 2;

            for (int i = 0; i < 5; i++) {
                coefsSpline[coefRow][coefCol] += C[i][j] * x[i];
            }
        }

        double[] yOut = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            int secondHalf = i < breaks[1] ? 0 : 1;
            yOut[i] = coefsSpline[secondHalf][0];
        }

        for (int i = 1; i < 4; i++) {
            for (int j = 0; j < arr.length; j++) {
                int secondHalf = j < breaks[1] ? 0 : 1;
                yOut[j] = yOut[j] * (j - breaks[1] * secondHalf) + coefsSpline[secondHalf][i];
            }
        }

        return yOut;
    }


}

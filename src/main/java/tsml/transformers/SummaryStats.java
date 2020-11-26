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

import experiments.data.DatasetLoading;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import utilities.InstanceTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;

import java.io.File;
import java.util.ArrayList;

/* simple Filter that just summarises the series
     * copyright: Anthony Bagnall

 * Global stats:
 * 		mean, variance, skewness, kurtosis, slope, min, max
 * */
public class SummaryStats implements Transformer {
    private int numMoments;
    private final String[] moment_names = { "mean", "variance", "skewness", "kurtosis", "slope", "min", "max" };

    public SummaryStats() {
        this(5);
    }

    public SummaryStats(int numM) {
        numMoments = Math.max(Math.min(numM, 5), 0); // no less than 0
    }

    public Instances determineOutputFormat(Instances inputFormat) {
        // Set up instances size and format.
        ArrayList<Attribute> atts = new ArrayList<>();
        String source = inputFormat.relationName();
        String name;
        for (int i = 0; i < moment_names.length; i++) {
            name = source + "Moment_" + moment_names[i];
            atts.add(new Attribute(name));
        }

        if (inputFormat.classIndex() >= 0) { // Classification set, set class
            // Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            ArrayList<String> vals = new ArrayList<>(target.numValues());
            for (int i = 0; i < target.numValues(); i++)
                vals.add(target.value(i));
            atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        Instances result = new Instances("Moments" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }

    @Override
    public Instance transform(Instance inst) {

        // 1. Get series:
        double[] d = InstanceTools.ConvertInstanceToArrayRemovingClassValue(inst);
        double[] moments = new double[numMoments + 2];

        double max = InstanceTools.max(inst);
        double min = InstanceTools.min(inst);
        double sum = InstanceTools.sum(inst);
        double sumSq = InstanceTools.sumSq(inst);

        moments[0] = sum / d.length;

        double totalVar = 0;
        double totalSkew = 0;
        double totalKur = 0;
        double p = 0;
        // Find variance
        if (numMoments > 0) {
            for (int j = 0; j < d.length; j++) {
                p = (d[j] - moments[0]) * (d[j] - moments[0]); // ^2
                totalVar += p;

                p *= (d[j] - moments[0]); // ^3
                totalSkew += p;

                p *= (d[j] - moments[0]); // ^4
                totalKur += p;
            }

            moments[1] = totalVar / (d.length - 1);
            double standardDeviation = Math.sqrt(moments[1]);
            moments[1] = standardDeviation;

            if (numMoments > 1) {
                double std3 = Math.pow(standardDeviation, 3);
                double skew = totalSkew / (std3);
                moments[2] = skew / d.length;
                if (numMoments > 2) {
                    double kur = totalKur / (std3 * standardDeviation);
                    moments[3] = kur / d.length;
                    if (numMoments > 3) {
                        // slope
                        moments[4] = standardDeviation != 0 ? InstanceTools.slope(inst, sum, sumSq) : 0;
                    }
                }
            }
        }
        double[] atts = new double[numMoments + 2 + (inst.classIndex() >= 0 ? 1 : 0)];
        for (int j = 0; j < numMoments; j++) {
            atts[j] = moments[j];
        }
        atts[numMoments] = min;
        atts[numMoments + 1] = max;

        if (inst.classIndex() >= 0)
            atts[atts.length - 1] = inst.classValue();

        return new DenseInstance(1.0, atts);
    }

    public static void main(String[] args) {

        String local_path = "D:\\Work\\Data\\Univariate_ts\\"; // Aarons local path for testing.
        String dataset_name = "ChinaTown";
        Instances train = DatasetLoading
                .loadData(local_path + dataset_name + File.separator + dataset_name + "_TRAIN.ts");
        Instances test = DatasetLoading
                .loadData(local_path + dataset_name + File.separator + dataset_name + "_TEST.ts");
        // Instances filter=new SummaryStats().process(test);
        SummaryStats m = new SummaryStats();
        Instances filter = m.transform(test);
        System.out.println(filter);
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        
        double[][] out = new double[inst.getNumDimensions()][];

        int i=0;
        for(TimeSeries ts : inst){
            TimeSeriesSummaryStatistics stats = new TimeSeriesSummaryStatistics(ts);
            //{ "mean", "variance", "skewness", "kurtosis", "slope", "min", "max" };
            int j=0;
            out[i] = new double[numMoments + 2];
            if (numMoments >= 0){
                out[i][j++] = stats.getMean();
                if (numMoments >= 1){
                    out[i][j++] = stats.getVariance();
                    if (numMoments >= 2){
                        out[i][j++] = stats.getSkew();
                        if (numMoments >= 3){
                            out[i][j++] = stats.getKurtosis();
                            if (numMoments >= 4){
                                out[i][j++] = stats.getSlope();
                            }
                        }
                    }
                }
            }
            out[i][j++] = stats.getMin();
            out[i][j++] = stats.getMax();
            i++;
        }
        return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
    }


}

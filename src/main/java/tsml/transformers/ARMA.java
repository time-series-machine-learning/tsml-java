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

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.ArrayList;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import utilities.InstanceTools;

/**
 * <!-- globalinfo-start --> Implementation of AR function as a Weka
 * SimpleBatchFilter Series to series transform independent of class value <!--
 * globalinfo-end --> <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 *  -L
 *  set the max lag.
 * </pre>
 * 
 * <!-- options-end -->
 *
 * 
 * author: Anthony Bagnall circa 2008. Reviewed and tidied up 2019 This should
 * not really be a batch filter, as it is series to series, but it makes the use
 * case simpler.
 */
public class ARMA implements Transformer {
        double[] ar;
        // Max number of AR terms to consider.
        public static int globalMaxLag = 25;
        // Defaults to 1/4 length of series
        public int maxLag = globalMaxLag;
        public boolean useAIC = false;

        public void setUseAIC(final boolean b) {
                useAIC = b;
        }

        public void setMaxLag(final int a) {
                maxLag = a;
        }

        @Override
        public Instances determineOutputFormat(final Instances inputFormat) {
                if (inputFormat.classIndex() >= 0) // Classification set, dont transform the target class!
                        maxLag = (inputFormat.numAttributes() - 1 > maxLag) ? maxLag : inputFormat.numAttributes() - 1;
                else
                        maxLag = (inputFormat.numAttributes() > maxLag) ? maxLag : inputFormat.numAttributes();
                // Set up instances size and format.
                final ArrayList<Attribute> atts = new ArrayList<>();
                String name;
                for (int i = 0; i < maxLag; i++) {
                        name = "ARMA_" + i;
                        atts.add(new Attribute(name));
                }
                if (inputFormat.classIndex() >= 0) { // Classification set, set class
                        // Get the class values as a fast vector
                        final Attribute target = inputFormat.attribute(inputFormat.classIndex());

                        final ArrayList<String> vals = new ArrayList<>(target.numValues());
                        for (int i = 0; i < target.numValues(); i++)
                                vals.add(target.value(i));
                        atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));

                }
                final Instances result = new Instances("ARMA" + inputFormat.relationName(), atts,
                                inputFormat.numInstances());
                if (inputFormat.classIndex() >= 0)
                        result.setClassIndex(result.numAttributes() - 1);
                return result;
        }

        @Override
        public Instance transform(Instance inst) {
                // 1. Get series
                double[] d = InstanceTools.ConvertInstanceToArrayRemovingClassValue(inst);
                // 2. Fit Autocorrelations
                final double[] pi = calculateValues(d);
                // 6. Stuff back into new Instances.

                int length = pi.length + (inst.classIndex() >= 0 ? 1 : 0);
                final Instance out = new DenseInstance(length);
                // Set class value.
                if (inst.classIndex() >= 0)
                        out.setValue(length - 1, inst.classValue());
                for (int k = 0; k < pi.length; k++) {
                        out.setValue(k, pi[k]);
                }
                return out;
        }

        private double[] calculateValues(double[] d) {
                double[] autos = ACF.fitAutoCorrelations(d, maxLag);
                // 3. Form Partials
                double[][] partials = PACF.formPartials(autos);
                // 4. Find best AIC. Could also use BIC?
                int best = maxLag;
                if (useAIC)
                        best = findBestAIC(autos, partials, maxLag, d);
                // 5. Find parameters
                final double[] pi = new double[maxLag];
                for (int k = 0; k < best; k++)
                        pi[k] = partials[k][best - 1];
                return pi;
        }

        public static double[] fitAR(final double[] d) {
                // 2. Fit Autocorrelations
                final double[] autos = ACF.fitAutoCorrelations(d, globalMaxLag);
                // 3. Form Partials
                final double[][] partials = PACF.formPartials(autos);
                // 4. Find bet AIC. Could also use BIC?
                final int best = findBestAIC(autos, partials, globalMaxLag, d);
                // 5. Find parameters
                final double[] pi = new double[globalMaxLag];
                for (int k = 0; k < best; k++)
                        pi[k] = partials[k][best - 1];
                return pi;
        }

        public static int findBestAIC(final double[] autoCorrelations, final double[][] partialCorrelations,
                        final int maxLag, final double[] d) {
                // need the variance of the series
                double sigma2;
                final int n = d.length;
                double var = 0, mean = 0;
                for (int i = 0; i < d.length; i++)
                        mean += d[i];
                for (int i = 0; i < d.length; i++)
                        var += (d[i] - mean) * (d[i] - mean);
                var /= (d.length - 1);
                double AIC = Double.MAX_VALUE;
                double bestAIC = Double.MAX_VALUE;
                int bestPos = 0;
                int i = 0;
                boolean found = false;
                while (i < maxLag && !found) {
                        sigma2 = 1;
                        for (int j = 0; j <= i; j++) {
                                sigma2 -= autoCorrelations[j] * partialCorrelations[j][i];
                                // System.out.println("\tStep ="+j+" incremental sigma ="+sigma2);
                        }
                        sigma2 *= var;
                        AIC = Math.log(sigma2);
                        i++;
                        AIC += ((double) 2 * (i + 1)) / n;
                        // System.out.println("LAG ="+i+" final sigma = "+sigma2+"
                        // log(sigma)="+Math.log(sigma2)+" AIC = "+AIC);
                        if (AIC == Double.NaN)
                                AIC = Double.MAX_VALUE;
                        if (AIC < bestAIC) {
                                bestAIC = AIC;
                                bestPos = i;
                        } else
                                found = true;
                }
                return bestPos;
        }

        public void setOptions(final String[] options) throws Exception {
                final String maxLagString = Utils.getOption('L', options);
                if (maxLagString.length() != 0)
                        this.maxLag = Integer.parseInt(maxLagString);
                else
                        this.maxLag = globalMaxLag;
        }

        @Override
        public TimeSeriesInstance transform(TimeSeriesInstance inst) {
                // could do this across all dimensions.
                double[][] out = new double[inst.getNumDimensions()][];
                int i = 0;
                for (TimeSeries ts : inst) {
                        out[i++] = calculateValues(ts.toValueArray());
                }

                // create a new output instance with the ACF data.
                return new TimeSeriesInstance(out, inst.getLabelIndex(), inst.getClassLabels());
        }

        /*
         * This function verifies the output is the same as from R The R code to perform
         * the ACF, ARMA and PACF comparison is in ....
         * 
         */
        /*
         * public static void testTransform(String path){ Instances
         * data=ClassifierTools.loadData(path+"ACFTest"); ACF acf=new ACF();
         * acf.setNormalized(false); PACF pacf=new PACF(); ARMA arma=new ARMA(); int
         * lag=10; acf.setMaxLag(lag); pacf.setMaxLag(lag); arma.setMaxLag(lag);
         * arma.setUseAIC(false); try{ Instances acfD=acf.process(data); Instances
         * pacfD=pacf.process(data); Instances armaD=arma.process(data); //Save first
         * case to file OutFile of=new OutFile(path+"ACFTest_JavaOutput.csv");
         * of.writeLine(",acf1,pacf1,arma"); for(int i=0;i<acfD.numAttributes()-1;i++)
         * of.writeLine("ar"+(i+1)+","+acfD.instance(0).value(i)+","+pacfD.instance(0).
         * value(i)+","+armaD.instance(0).value(i)); double[][]
         * partials=pacf.getPartials(); of.writeLine("\n\n"); for(int
         * i=0;i<partials.length;i++){ of.writeString("\n"); for(int
         * j=0;j<partials[i].length;j++) of.writeString(partials[i][j]+",");
         * 
         * }
         * 
         * } catch(Exception e){ System.out.println("Exception caught, exit "+e);
         * e.printStackTrace(); System.exit(0); } } public static void main(String[]
         * args){
         * 
         * testTransform("C:\\Users\\ajb\\Dropbox\\TSC Problems\\TestData\\");
         * System.exit(0); //Debug code to test. ARMA ar = new ARMA();
         * ar.setUseAIC(false);
         * 
         * //Generate a model double[][] paras={{0.5},{0.7}}; int n=100; int cases=1; //
         * double[][]
         * paras={{1.3532,0.4188,-1.2153,0.3091,0.1877,-0.0876,0.0075,0.0004}, //
         * {1.0524,0.9042,-1.2193,0.0312,0.263,-0.0567,-0.0019} };
         * 
         * //Generate a series
         * 
         * //Fit and compare paramaters without AIC
         * 
         * //Fit using AIC and compare again
         * 
         * 
         * }
         */

}

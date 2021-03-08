/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
package machine_learning.classifiers;

import experiments.data.DatasetLoading;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;
import tsml.classifiers.MultiThreadable;
import tsml.transformers.ROCKET;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

import static utilities.InstanceTools.resampleTrainAndTestInstances;

/**
 * Ridge classification with cross-validation to select the alpha value.
 *
 * Based on RidgeClassifierCV from sklearn.
 * https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html
 *
 * @author Matthew Middlehurst
 */
public class RidgeClassifierCV extends AbstractClassifier implements MultiThreadable {

    //alphas used in sktime ROCKET
    private final double[] alphas = {1.00000000e-03, 4.64158883e-03, 2.15443469e-02, 1.00000000e-01,
            4.64158883e-01, 2.15443469e+00, 1.00000000e+01, 4.64158883e+01, 2.15443469e+02, 1.00000000e+03};

    private INDArray coefficients;
    private double[] intercept;

    private int numThreads = 1;

    private double bestScore = -999999;

    public double getBestScore() { return bestScore; }

    @Override
    public void enableMultiThreading(int numThreads) {
        this.numThreads = numThreads;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        if (instances.classIndex() != instances.numAttributes() - 1)
            throw new Exception("Class attribute must be the final index.");

        //Set to OMP_NUM_THREADS=1 for single thread run
        String value = System.getenv("OMP_NUM_THREADS");
        if (value == null && numThreads != Runtime.getRuntime().availableProcessors())
            throw new Exception("RidgeClassifierCV: OMP_NUM_THREADS environmental variable not set. Set it to the " +
                    "number of threads you wish to use or set numThreads to " +
                    "Runtime.getRuntime().availableProcessors()");
        if (value != null && Integer.parseInt(value) != numThreads)
            throw new Exception("RidgeClassifierCV: OMP_NUM_THREADS environmental variable and numThreads do not " +
                    "match.");

        bestScore = -999999;

        double[][] data = new double[instances.numInstances()][instances.numAttributes()-1];
        for (int i = 0; i < data.length; i++) {
            Instance inst = instances.get(i);
            for (int n = 0; n < data[i].length; n++) {
                data[i][n] = inst.value(n);
            }
        }

        double[][] labels;
        if (instances.numClasses() > 2){
            labels = new double[data.length][instances.numClasses()];
            for (int i = 0; i < data.length; i++) {
                Instance inst = instances.get(i);
                for (int n = 0; n < labels[i].length; n++) {
                    if (inst.classValue() == n){
                        labels[i][n] = 1;
                    }
                    else{
                        labels[i][n] = -1;
                    }
                }
            }
        }
        else{
            labels = new double[data.length][1];
            for (int i = 0; i < data.length; i++) {
                if (instances.get(i).classValue() == 1){
                    labels[i][0] = 1;
                }
                else{
                    labels[i][0] = -1;
                }
            }
        }

        double[] xOffset = new double[data[0].length];
        double[] yOffset = new double[labels[0].length];
        double[] xScale = new double[data[0].length];
        preprocessData(data, labels, xOffset, yOffset, xScale);

        //original uses SVD when no. instances > no. attributes
        INDArray matrix = Nd4j.create(data);
        INDArray q = matrix.mmul(matrix.transpose());
        INDArray eigvals = Eigen.symmetricGeneralizedEigenvalues(q);
        INDArray qt_y = q.transpose();
        qt_y = qt_y.mmul(Nd4j.create(labels));

        INDArray bestCoef = null;
        for (double alpha : alphas){
            double[] w = new double[(int)eigvals.size(0)];
            for (int i = 0; i < w.length; i++){
                w[i] = 1./(eigvals.getDouble(i) + alpha);
            }

            double[][] p = new double[1][data.length];
            Arrays.fill(p[0], Math.sqrt(data.length)/data.length);
            INDArray sw = Nd4j.create(p);
            double[] k = sw.mmul(q).toDoubleVector();
            for (int i = 0 ; i < k.length; i++) k[i] = Math.abs(k[i]);
            int idx = argmax(k);
            w[idx] = 0;

            double[][] d = new double[w.length][(int)qt_y.size(1)];
            for (int i = 0; i < d.length; i++){
                for (int n = 0; n < d[i].length; n++){
                    d[i][n] = w[i] * qt_y.getDouble(i,n);
                }
            }

            INDArray coefs = q.mmul(Nd4j.create(d));

            double[] sums = new double[w.length];
            for (int i = 0; i < w.length; i++){
                for (int n = 0; n < q.size(0); n++){
                    sums[n] += w[i] * Math.pow(q.getDouble(n,i), 2);
                }
            }

            double e = 0;
            for (int i = 0; i < sums.length; i++){
                for (int n = 0; n < coefs.size(1); n++){
                    e += Math.pow(coefs.getDouble(i,n) / sums[i], 2);
                }
            }
            e /= sums.length * coefs.size(1);
            e = 1 - e;

            if (e > bestScore){
                bestScore = e;
                bestCoef = coefs;
            }
        }

        INDArray a = bestCoef.transpose().mmul(matrix);
        double[][] b = a.size(0) == 1 ? new double[][]{ a.toDoubleVector() } : a.toDoubleMatrix();
        for (int i = 0; i < b.length; i++){
            for (int n = 0; n < b[i].length; n++){
                b[i][n] /= xScale[n];
            }
        }

        double[][] c = new double[][]{ xOffset };
        coefficients = Nd4j.create(b).transpose();
        INDArray d = Nd4j.create(c).mmul(coefficients);
        intercept = new double[yOffset.length];
        for (int i = 0; i < intercept.length; i++){
            intercept[i] = yOffset[i] - d.getDouble(i);
        }
    }

    @Override
    public double classifyInstance(Instance inst){
        double[][] data = new double[1][(int)coefficients.size(0)];
        for (int i = 0; i < data[0].length; i++) {
            data[0][i] = inst.value(i);
        }

        double[] x = Nd4j.create(data).mmul(coefficients).toDoubleVector();
        for (int i = 0; i < intercept.length; i++) {
            x[i] += intercept[i];
        }

        return x.length > 1 ? argmax(x) : (x[0] > 0 ? 1 : 0);
    }

    private void preprocessData(double[][] data, double[][] labels, double[] xOffset, double[] yOffset,
                                double[] xScale){
        for (int i = 0; i < data.length; i++) {
            for (int n = 0; n < data[i].length; n++) {
                xOffset[n] += data[i][n];
            }

            for (int n = 0; n < labels[i].length; n++) {
                yOffset[n] += labels[i][n];
            }
        }

        for (int i = 0; i < xOffset.length; i++){
            xOffset[i] /= data.length;
        }

        for (int i = 0; i < yOffset.length; i++){
            yOffset[i] /= labels.length;
        }

        for (int i = 0; i < data.length; i++) {
            for (int n = 0; n < data[i].length; n++) {
                data[i][n] -= xOffset[n];
            }

            for (int n = 0; n < labels[i].length; n++) {
                labels[i][n] -= yOffset[n];
            }
        }

        for (double[] row : data) {
            for (int n = 0; n < row.length; n++) {
                xScale[n] += row[n] * row[n];
            }
        }

        for (int i = 0; i < xOffset.length; i++){
            xScale[i] = Math.sqrt(xScale[i]);
            if (xScale[i] == 0) xScale[i] = 1;
        }

        for (int i = 0; i < data.length; i++) {
            for (int n = 0; n < data[i].length; n++) {
                data[i][n] /= xScale[n];
            }
        }
    }

    private int argmax(double[] arr){
        double max = -999999;
        int idx = -1;
        for (int i = 0; i < arr.length; i++){
            if (arr[i] > max){
                max = arr[i];
                idx = i;
            }
        }
        return idx;
    }

    public static void main(String[] args) throws Exception {
        int fold = 0;

        //Minimum working example
        String dataset = "ItalyPowerDemand";
        Instances train = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset
                + "\\" + dataset + "_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset
                + "\\" + dataset + "_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        RidgeClassifierCV c = new RidgeClassifierCV();
        double accuracy;

        ROCKET r = new ROCKET();
        Instances tTrain = r.fitTransform(train);

        c.buildClassifier(tTrain);

        accuracy = ClassifierTools.accuracy(r.transform(test), c);
        System.out.println(accuracy);
    }
}

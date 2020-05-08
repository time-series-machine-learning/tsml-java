/* Copyright (C) 2019 Chang Wei Tan
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. */
package tsml.classifiers.distance_based;

import experiments.data.DatasetLoading;
import tsml.classifiers.legacy.elastic_ensemble.fast_elastic_ensemble.utils.SequenceStatsCache;
import tsml.classifiers.legacy.elastic_ensemble.Efficient1NN;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

/**
 * This is a method introduced in the paper
 * FastEE: Fast Ensembles of Elastic Distances for Time Series Classification.
 * It builds a nearest neighbour table that can be used to learn the optimal parameter for each distance
 * measure.
 * Once this table is built, learning the optimal parameter can be done at one pass.
 * Please refer to the paper for more details about this table.
 *
 * @author Chang Wei (chang.tan@monash.edu)
 */

public class FastElasticEnsemble extends ElasticEnsemble {
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "C. Tan, F. Petitjean and G. Webb");
        result.setValue(TechnicalInformation.Field.TITLE, "FastEE: Fast Ensembles of Elastic Distances for Time Series Classification");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");
        result.setValue(TechnicalInformation.Field.VOLUME, "XXX");
        result.setValue(TechnicalInformation.Field.NUMBER, "XXX");

        result.setValue(TechnicalInformation.Field.PAGES, "XXX");
        result.setValue(TechnicalInformation.Field.YEAR, "2019");
        return result;
    }

    public FastElasticEnsemble() {
        this.classifiersToUse = ConstituentClassifiers.values();
    }

    public FastElasticEnsemble(ConstituentClassifiers[] classifiersToUse) {
        this.classifiersToUse = classifiersToUse;
    }

    public FastElasticEnsemble(String resultsDir, String datasetName, int resampleId) {
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
        this.classifiersToUse = ConstituentClassifiers.values();
        this.buildFromFile = true;
    }

    public FastElasticEnsemble(String resultsDir, String datasetName, int resampleId, ConstituentClassifiers[] classifiersToUse) {
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
        this.classifiersToUse = classifiersToUse;
        this.buildFromFile = true;
    }

/*    @Override
    public double getTrainAcc() {
        if (this.ensembleCvAcc != -1 && this.ensembleCvPreds != null) {
            return this.ensembleCvAcc;
        }

        this.getTrainPreds();
        return this.ensembleCvAcc;
    }
*/
    @Override
    public void buildClassifier(Instances train) throws Exception {
        long t1= System.nanoTime();
        this.train = train;
        this.derTrain = null;
        usesDer = false;

        this.classifiers = new Efficient1NN[this.classifiersToUse.length];
        this.cvAccs = new double[classifiers.length];
        this.cvPreds = new double[classifiers.length][this.train.numInstances()];

        for (int c = 0; c < classifiers.length; c++) {
            classifiers[c] = getClassifier(this.classifiersToUse[c]);
            if (isDerivative(this.classifiersToUse[c])) {
                usesDer = true;
            }
        }

        if (usesDer) {
            this.derTrain = df.transform(train);
        }

        if (buildFromFile) {
            File existingTrainOut;
            Scanner scan;
            int paramId;
            double cvAcc;
            for (int c = 0; c < classifiers.length; c++) {
                existingTrainOut = new File(this.resultsDir + classifiersToUse[c] + "/Predictions/" + datasetName + "/trainFold" + this.resampleId + ".csv");
                if (!existingTrainOut.exists()) {
                    throw new Exception("Error: training file doesn't exist for " + existingTrainOut.getAbsolutePath());
                }
                scan = new Scanner(existingTrainOut);
                scan.useDelimiter("\n");
                scan.next();//header
                paramId = Integer.parseInt(scan.next().trim().split(",")[0]);
                cvAcc = Double.parseDouble(scan.next().trim().split(",")[0]);

                for (int i = 0; i < train.numInstances(); i++) {
                    this.cvPreds[c][i] = Double.parseDouble(scan.next().split(",")[1]);
                }

                scan.close();
                if (isDerivative(classifiersToUse[c])) {
                    if (!isFixedParam(classifiersToUse[c])) {
                        classifiers[c].setParamsFromParamId(derTrain, paramId);
                    }
                    classifiers[c].buildClassifier(derTrain);
                } else {
                    if (!isFixedParam(classifiersToUse[c])) {
                        classifiers[c].setParamsFromParamId(train, paramId);
                    }
                    classifiers[c].buildClassifier(train);
                }
                cvAccs[c] = cvAcc;
            }
        } else {
            double[] cvAccAndPreds;
            for (int c = 0; c < classifiers.length; c++) {
                if (writeToFile) {
                    classifiers[c].setFileWritingOn(this.resultsDir, this.datasetName, this.resampleId);
                }
                if (isFixedParam(classifiersToUse[c])) {
                    if (isDerivative(classifiersToUse[c])) {
                        cvAccAndPreds = classifiers[c].loocv(derTrain);
                    } else {
                        cvAccAndPreds = classifiers[c].loocv(train);
                    }
                } else if (isDerivative(classifiersToUse[c])) {
                    cvAccAndPreds = classifiers[c].fastParameterSearch(derTrain);
                } else {
                    cvAccAndPreds = classifiers[c].fastParameterSearch(train);
                }

                cvAccs[c] = cvAccAndPreds[0];
                for (int i = 0; i < train.numInstances(); i++) {
                    this.cvPreds[c][i] = cvAccAndPreds[i + 1];
                }
            }

/*
            if (this.writeEnsembleTrainingFile) {
                StringBuilder output = new StringBuilder();

                double[] ensembleCvPreds = this.getTrainPreds();

                output.append(train.relationName()).append(",FastEE,train\n");
                output.append(this.getParameters()).append("\n");
                output.append(trainResults.getAcc()).append("\n");

                for (int i = 0; i < train.numInstances(); i++) {
                    output.append(train.instance(i).classValue()).append(",").append(ensembleCvPreds[i]).append("\n");
                }

                FileWriter fullTrain = new FileWriter(this.ensembleTrainFilePathAndName);
                fullTrain.append(output);
                fullTrain.close();
            }
   */     
        }
        trainResults.setBuildTime(System.nanoTime() - t1);
    }

    // classify instance with lower bounds
    public double classifyInstance(final Instance instance, final int queryIndex, final SequenceStatsCache cache) throws Exception{
        if(classifiers==null){
            throw new Exception("Error: classifier not built");
        }
        Instance derIns = null;
        if(this.usesDer){
            Instances temp = new Instances(derTrain,1);
            temp.add(instance);
            temp = df.transform(temp);
            derIns = temp.instance(0);
        }

        double bsfVote = -1;
        double[] classTotals = new double[train.numClasses()];
        ArrayList<Double> bsfClassVal = null;

        double pred;
        this.previousPredictions = new double[this.classifiers.length];

        for(int c = 0; c < classifiers.length; c++){
            if(isDerivative(classifiersToUse[c])){
                pred = classifiers[c].classifyInstance(derTrain, derIns, queryIndex, cache);
            }else{
                pred = classifiers[c].classifyInstance(train, instance, queryIndex, cache);
            }
            previousPredictions[c] = pred;

            try{
                classTotals[(int)pred] += cvAccs[c];
            }catch(Exception e){
                System.out.println("cv accs "+cvAccs.length);
                System.out.println(pred);
                throw e;
            }

            if(classTotals[(int)pred] > bsfVote){
                bsfClassVal = new ArrayList<>();
                bsfClassVal.add(pred);
                bsfVote = classTotals[(int)pred];
            }else if(classTotals[(int)pred] == bsfVote){
                bsfClassVal.add(pred);
            }
        }

        if(bsfClassVal.size()>1){
            return bsfClassVal.get(new Random(46).nextInt(bsfClassVal.size()));
        }
        return bsfClassVal.get(0);
    }

    private String getClassifierInfo() {
        StringBuilder st = new StringBuilder();
        st.append("FastEE using:\n");
        st.append("=====================\n");
        for (int c = 0; c < classifiers.length; c++) {
            st.append(classifiersToUse[c]).append(" ").append(classifiers[c].getClassifierIdentifier()).append(" ").append(cvAccs[c]).append("\n");
        }
        return st.toString();
    }

    @Override
    public String toString() {
        return super.toString() + "\n" + this.getClassifierInfo();
    }

    public static void main(String[] args) throws Exception {
        FastElasticEnsemble ee = new FastElasticEnsemble();
        Instances train = DatasetLoading.loadDataNullable("C:/Users/cwtan/workspace/Dataset/TSC_Problems/ArrowHead/ArrowHead_TRAIN");
        Instances test = DatasetLoading.loadDataNullable("C:/Users/cwtan/workspace/Dataset/TSC_Problems/ArrowHead/ArrowHead_TEST");
        ee.buildClassifier(train);

        SequenceStatsCache cache = new SequenceStatsCache(test, test.numAttributes() - 1);

        System.out.println("Train Acc: " + ee.trainResults.getAcc());
        int correct = 0;
        for (int i = 0; i < test.numInstances(); i++) {
            double actual = test.instance(i).classValue();
            double pred = ee.classifyInstance(test.instance(i), i, cache);
            if (actual == pred) {
                correct++;
            }
        }
        System.out.println("Test Acc: " + (double) correct / test.numInstances());
        System.out.println("Test Acc -- correct: " + correct + "/" + test.numInstances());
    }
}

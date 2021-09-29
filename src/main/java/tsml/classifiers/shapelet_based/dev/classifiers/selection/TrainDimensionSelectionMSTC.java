package tsml.classifiers.shapelet_based.dev.classifiers.selection;

import evaluation.storage.ClassifierResults;
import experiments.ExperimentsTS;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.Collections;

public class TrainDimensionSelectionMSTC extends DimensionSelectionMSTC {

    protected int k;
    private String TRAIN_ALG = "STC";

    public TrainDimensionSelectionMSTC(ExperimentsTS.ExperimentalArguments exp, MSTC.ShapeletParams params){
        super(exp,params);
    }

    protected ArrayList<DimensionResult> getDimensionResults(TimeSeriesInstances data) throws Exception{
        ArrayList<DimensionResult> dimensionResults = new ArrayList<DimensionResult>();
        for (int i=0;i<this.numDimensions;i++){
            ClassifierResults results = new ClassifierResults(this.exp.resultsWriteLocation + "STC/Predictions/" +
                    this.exp.datasetName + "Dimension" + (i+1) + "/trainFold" + this.exp.foldId + ".csv");
            dimensionResults.add(new DimensionResult(i,results.getAcc()));

        }
        return dimensionResults;
    }

    int[] getIndexes(TimeSeriesInstances data) throws Exception{


        ArrayList<DimensionResult> dimensionResults = getDimensionResults(data);
        Collections.sort(dimensionResults);
        this.k = getElbow(dimensionResults)+1;
        System.out.println("Selected " + this.k + " of " + this.numDimensions);
        dimensionResults.subList(k,dimensionResults.size()).clear();
        return dimensionResults.stream().mapToInt(i->i.dimensionIndex).toArray();

    }


    protected int getElbow(ArrayList<TrainDimensionSelectionMSTC.DimensionResult> dimensionResults){
       int  nPoints = dimensionResults.size();

        DimensionResult firstPoint = dimensionResults.get(0);
        DimensionResult lastPoint = dimensionResults.get(nPoints-1);

        double[] lineVec = {nPoints-1, lastPoint.accuracy- firstPoint.accuracy};
        double lineVecSumSqrt = Math.sqrt(lineVec[0]*lineVec[0]+lineVec[1]*lineVec[1]);
        double[] lineVecNorm = {lineVec[0] / lineVecSumSqrt, lineVec[1] /lineVecSumSqrt};
        double[][] vecFromFirst = new double[nPoints][2];
        double[][] scalar = new double[nPoints][2];
        double[][] vecFromFirstParallel = new double[nPoints][2];
        double[][] vec_to_line = new double[nPoints][2];

        double[] scalarProd = new double[nPoints];
        double[] distToLine = new double[nPoints];
        int index = 0;
        double maxDistToLine = -9999;
        for (int i=0;i<nPoints;i++){
            vecFromFirst[i][0] = i;
            vecFromFirst[i][1] = dimensionResults.get(i).accuracy - firstPoint.accuracy;
            scalar[i][0] = vecFromFirst[i][0] * lineVecNorm[0];
            scalar[i][1] = vecFromFirst[i][1] * lineVecNorm[1];
            scalarProd[i] =  scalar[i][0] + scalar[i][1];
            vecFromFirstParallel[i][0] = scalarProd[i] * lineVecNorm[0];
            vecFromFirstParallel[i][1] = scalarProd[i] * lineVecNorm[1];
            vec_to_line[i][0] = vecFromFirst[i][0] - vecFromFirstParallel[i][0];
            vec_to_line[i][1] = vecFromFirst[i][1] - vecFromFirstParallel[i][1];
            distToLine[i] = Math.sqrt(  vec_to_line[i][0]*vec_to_line[i][0] + vec_to_line[i][1]*vec_to_line[i][1] );
            if (distToLine[i]>maxDistToLine){
                maxDistToLine = distToLine[i];
                index = i;
            }
        }
        return index;
    }

    class DimensionResult implements Comparable<DimensionResult>{
        Integer dimensionIndex;
        double accuracy;
        public DimensionResult(int dimensionIndex, double accuracy){
            this.dimensionIndex = dimensionIndex;
            this.accuracy = accuracy;
        }

        @Override
        public int compareTo(DimensionResult other) {
            return (Double.compare(other.accuracy, this.accuracy));
        }
    }
}

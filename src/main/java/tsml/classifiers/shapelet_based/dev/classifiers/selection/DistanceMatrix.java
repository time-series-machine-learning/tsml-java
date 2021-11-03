package tsml.classifiers.shapelet_based.dev.classifiers.selection;

import experiments.ExperimentsTS;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.List;

public class DistanceMatrixMSTC extends ElbowSelection {

    int numClasses;

    public DistanceMatrixMSTC(int numClasses, ExperimentsTS.ExperimentalArguments exp, MSTC.ShapeletParams params){
        super(numClasses, exp,params);
    }

    protected  ArrayList<CentroidMatrixItem> getCentroidMatrix(TimeSeriesInstances data){
        ArrayList<CentroidMatrixItem> centroidMatrix = new ArrayList<CentroidMatrixItem>();
        this.numClasses = data.numClasses();
        for (int k=0;k<this.numDimensions;k++){
            TimeSeriesInstances classInstances = new TimeSeriesInstances(data.getHSliceArray(new int[]{k}),data.getClassIndexes(), data.getClassLabels());
            List<TimeSeriesInstances> byClass = classInstances.getInstsByClass();
            for (int i=0;i<byClass.size();i++){
                TimeSeriesInstances instances = byClass.get(i);
                double[] cen = new double[instances.getMinLength()];
                for (TimeSeriesInstance instance: instances){
                    for (int j=0;j<cen.length;j++){
                        cen[j] += instance.get(0).get(j);
                    }
                }
                for (int j=0;j<cen.length;j++){
                    cen[j] /= (double)instances.numInstances();
                }
                centroidMatrix.add(new CentroidMatrixItem(i,k,cen));

            }
        }
        return centroidMatrix;
    }
    protected ArrayList<DistanceMatrixItem> getDistanceMatrix(ArrayList<CentroidMatrixItem> centroidMatrix){
        ArrayList<DistanceMatrixItem> distanceMatrix = new ArrayList<DistanceMatrixItem>();

        for (int i=0;i<this.numClasses-1;i++){
            for (int j=i+1;j<this.numClasses;j++){
                for (int k=0;k<this.numDimensions;k++){
                    final int ii = i;
                    final int jj = j;
                    final int kk = k;
                    CentroidMatrixItem cen1 = centroidMatrix.stream()
                            .filter(cmi -> cmi.classIndex == ii && cmi.dimensionIndex == kk)
                            .findFirst()
                            .orElse(null);

                    CentroidMatrixItem cen2 = centroidMatrix.stream()
                            .filter(cmi -> cmi.classIndex == jj && cmi.dimensionIndex == kk)
                            .findFirst()
                            .orElse(null);

                    if (cen1 != null && cen2 != null){
                        double d = dist(cen1.centroid, cen2.centroid);
                        distanceMatrix.add(new DistanceMatrixItem(ii,jj,kk,d));
                    }

                }
            }

        }
        return distanceMatrix;
    }

    protected ArrayList<DimensionResult> getDimensionResults( ArrayList<DistanceMatrixItem> distanceMatrix){
        ArrayList<DimensionResult> dimensionResults = new ArrayList<DimensionResult>();
        for (int k=0;k<this.numDimensions;k++){
            final int kk = k;
            double sum = distanceMatrix.stream()
                    .filter(dmi -> dmi.dimensionIndex ==kk)
                    .mapToDouble(dmi -> dmi.score)
                    .sum();
            dimensionResults.add(new DimensionResult(k,sum));
        }
        return dimensionResults;

    }

    @Override
    protected ArrayList<DimensionResult> getDimensionResults(TimeSeriesInstances data) throws Exception {
        this.numClasses = data.numClasses();

        ArrayList<CentroidMatrixItem> centroidMatrix = getCentroidMatrix(data);
        ArrayList<DistanceMatrixItem> distanceMatrix = getDistanceMatrix(centroidMatrix);
       return getDimensionResults(distanceMatrix);

//        int dmSize = this.numDimensions * (this.numClasses * (this.numClasses-1) /2);

    }


    class CentroidMatrixItem{
        int classIndex;
        int dimensionIndex;
        double[] centroid;

        public CentroidMatrixItem(int classIndex, int dimensionIndex, double[] centroid){
            this.classIndex = classIndex;
            this.dimensionIndex = dimensionIndex;
            this.centroid = centroid;
        }
    }

    class DistanceMatrixItem implements Comparable<DistanceMatrixItem>{
        int classIndex1;
        int classIndex2;
        int dimensionIndex;
        double score;
        public DistanceMatrixItem(int classIndex1, int classIndex2, int dimensionIndex, double score){
            this.classIndex1 = classIndex1;
            this.classIndex2 = classIndex2;
            this.dimensionIndex = dimensionIndex;
            this.score = score;
        }
        @Override
        public int compareTo(DistanceMatrixItem other) {
            return (Double.compare(other.score, this.score));
        }

    }

    double dist(double[] a, double[] b){
        double sum = 0;
        for (int i=0;i<a.length;i++){
            sum += (b[i] - a[i])*(b[i] - a[i]);
        }
        return Math.sqrt(sum);
    }
}

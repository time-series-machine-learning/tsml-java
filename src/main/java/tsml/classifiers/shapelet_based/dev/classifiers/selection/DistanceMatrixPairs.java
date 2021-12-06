package tsml.classifiers.shapelet_based.dev.classifiers.selection;

import experiments.Experiments;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class DistanceMatrixPairs extends DistanceMatrix {

    int numClasses;



    @Override
    int[] getIndexes(TimeSeriesInstances data) throws Exception{
        this.numClasses = data.numClasses();
        ArrayList<CentroidMatrixItem> centroidMatrix = getCentroidMatrix(data);
        ArrayList<DistanceMatrixItem> distanceMatrix = getDistanceMatrix(centroidMatrix);
        boolean[] included = new boolean[this.numDimensions];
        for (int i=0;i<this.numClasses-1;i++) {
            for (int j = i + 1; j < this.numClasses; j++) {
                final int ii = i;
                final int jj = j;
                List<DistanceMatrixItem> items  = distanceMatrix.stream()
                        .filter(dmi -> dmi.classIndex1 == ii && dmi.classIndex2 == jj).collect(Collectors.toList());

                double max = -9999;
                int index = -1;
                for (DistanceMatrixItem item: items){
                    if (item.score > max){
                        max = item.score;
                        index = item.dimensionIndex;
                    }
                }

                included[index] = true;

            }
        }
        int numIncluded = 0;
        for (int i=0;i<included.length;i++){
            if (included[i]) numIncluded++;
        }
        int[] indexes = new int[numIncluded];
        int j =0;
        for (int i=0;i<included.length;i++){
            if (included[i]){
                indexes[j] = i;
                j++;
            }
        }
        return indexes;
    }
}

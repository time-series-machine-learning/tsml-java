package tsml.classifiers.shapelet_based.dev.classifiers.selection;

import experiments.Experiments;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.filter.RandomFilter;
import tsml.classifiers.shapelet_based.dev.filter.RandomFilterBySeries;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ShapeletSeriesDimensionSelection extends ElbowSelection {


    public ShapeletSeriesDimensionSelection(Experiments.ExperimentalArguments exp, MSTC.ShapeletParams params){
        super( exp,params);
        this.setAbleToEstimateOwnPerformance(true);
    }

    @Override
    protected ArrayList<DimensionResult> getDimensionResults(TimeSeriesInstances data) throws Exception {
        return null;
    }

    int[] getIndexes(TimeSeriesInstances instances) throws Exception{
       // ArrayList<DimensionResult> dimensionResults = new ArrayList<DimensionResult>();
        ArrayList<ShapeletGroup> shapeletGroup = new ArrayList<ShapeletGroup>();
        RandomFilter randomFilter = new RandomFilter();
        randomFilter.setMinuteLimit(30);
        MSTC.ShapeletParams p = new MSTC.ShapeletParams(params);
        p.maxIterations = 10000;
        p.allowZeroQuality = true;
        p.k = 100;
        int[][] indexes = new int[instances.numClasses()][];
        for (int j=0;j<instances.numClasses();j++) {
            final int finalJ = j;
            indexes[j] = IntStream.range(0, instances.numInstances())
                    .filter(index -> instances.get(index).getLabelIndex() == finalJ)
                    .toArray();
        }

        for (int i=0;i<instances.getMaxNumDimensions();i++){
            for (int j=0;j<instances.numClasses();j++){
                final int finalJ = j;

                ShapeletGroup sg = new ShapeletGroup(i,j);

                sg.setShapelets(randomFilter.findShapelets(p,instances,indexes[j],i));
                System.out.println(sg);
                shapeletGroup.add(sg);

            }

        }

        Set<Integer> selections = new HashSet<>();
        for (int j=0;j<instances.numClasses();j++){
            final int localJ = j;
            List<DimensionResult> elements = shapeletGroup
                    .stream()
                    .filter(sg -> sg.classIndex == localJ)
                    .map(sg -> new DimensionResult(sg.dimension,sg.quality))
                    .collect(Collectors.toList());
            Collections.sort(elements);
            int elbow = getElbow((ArrayList<DimensionResult>) elements);
            for (int i=0;i<elbow+1;i++){
                selections.add(elements.get(i).dimensionIndex);
            }

        }
        System.out.println(selections);

        return selections.stream().mapToInt(i->i).toArray();
    }

    class ShapeletGroup implements Comparable<ShapeletGroup>{
        private List<ShapeletMV> shapelets;
        private int dimension,classIndex;
        private double quality;
        public ShapeletGroup( int dimension,int classIndex){
            shapelets = new ArrayList<>();
            this.dimension = dimension;
            this.classIndex = classIndex;
        }
        public void setQuality(double q){
            this.quality = q;
        }
        public void setShapelets( List<ShapeletMV> shapelets){
            this.shapelets = shapelets;
            this.setQuality(this.shapelets.stream().mapToDouble(ShapeletMV::getQuality).average().orElse(0));
        }

        public String toString(){
            return "Dimension " + dimension + " class " + classIndex + " quality " + quality;
        }

        public int compareTo(ShapeletGroup other) {
            return (Double.compare(other.quality, this.quality));
        }
    }


}

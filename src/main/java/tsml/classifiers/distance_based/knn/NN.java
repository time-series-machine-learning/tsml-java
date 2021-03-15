package tsml.classifiers.distance_based.knn;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.ed.EDistance;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import tsml.classifiers.distance_based.utils.collections.CollectionUtils;
import tsml.classifiers.distance_based.utils.collections.lists.IndexList;
import tsml.classifiers.distance_based.utils.collections.lists.UnorderedArrayList;
import tsml.classifiers.distance_based.utils.collections.pruned.PrunedMultimap;
import tsml.classifiers.distance_based.utils.system.random.RandomUtils;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.ArrayUtilities;

import java.util.*;

public class NN extends BaseClassifier {

    public NN() {
        setK(1);
        setDistanceMeasure(new EDistance());
        setEarlyAbandonDistances(true);
    }
    
    private TimeSeriesInstances trainData;
    private int k;
    private DistanceMeasure distanceMeasure;
    private boolean earlyAbandonDistances;
    private boolean instanceExaminationReordering;
    
    @Override public void buildClassifier(final TimeSeriesInstances trainData) throws Exception {
        super.buildClassifier(trainData);
        this.trainData = trainData;
        if(getEstimateOwnPerformance()) {
            // conduct a partial or full loocv
            // maintain list of the left out insts, i.e. evaluation points
            List<NeighbourSearch> leftOut = new ArrayList<>();
            // maintain a list of the candidates not examined as potential neighbours by each left out's search
            List<Integer> remainingNeighbours = new UnorderedArrayList<>(new IndexList(trainData.numInstances()));
            
            while(!remainingNeighbours.isEmpty()) {
                final int index = RandomUtils.choiceIndex(remainingNeighbours.size(), getRandom());
                final Integer instIndex = remainingNeighbours.remove(index);
                // the chosen inst is now left out, so create a corresponding search
                final NeighbourSearch search = new NeighbourSearch(trainData.get(instIndex), instIndex);
                // add all previous seen / left out insts to the search
                for(NeighbourSearch other : leftOut) {
                    final double otherLimit = other.getLimit();
                    final double thisLimit = search.getLimit();
//                    final double distance = search.add(other.getInstIndex(), Math.max(thisLimit, otherLimit));
//                    other.add(instIndex, distance);
                }
                leftOut.add(search);
            }
        }
    }

    @Override public double[] distributionForInstance(final TimeSeriesInstance inst) throws Exception {
        final NeighbourSearch search = new NeighbourSearch(inst, -1);
        for(Double distance : search) {
            // neighbour examined
        }
        return search.predict();
    }

    public boolean isInstanceExaminationReordering() {
        return instanceExaminationReordering;
    }

    public void setInstanceExaminationReordering(final boolean instanceExaminationReordering) {
        this.instanceExaminationReordering = instanceExaminationReordering;
    }

    private class NeighbourSearch implements Iterable<Double> {
        private final PrunedMultimap<Double, Integer> neighbourMap;
        private double limit = Double.POSITIVE_INFINITY;
        private final TimeSeriesInstance inst;
        private double[] distribution;
        private final int instIndex;

        public int getInstIndex() {
            return instIndex;
        }

        public NeighbourSearch(final TimeSeriesInstance inst, final int instIndex) {
            this.inst = Objects.requireNonNull(inst);
            this.instIndex = instIndex;
            neighbourMap = PrunedMultimap.asc();
            neighbourMap.setSoftLimit(k);
        }
        
        public double add(int i) {
            if(i == instIndex) {
                return 0;
            }
            final TimeSeriesInstance neighbour = trainData.get(i);
            final double distance = distanceMeasure.distance(inst, neighbour, limit);
            return add(i, distance);
        }
        
        public double add(int i, double distance) {
            if(i == instIndex) {
                return 0;
            }
            final boolean nearestNeighbour = neighbourMap.put(distance, i);
            limit = neighbourMap.lastKey();
            if(nearestNeighbour) {
                distribution = null;
            }
            return distance;
        }

        public double getLimit() {
            return limit;
        }

        public void setLimit(double limit) {
            this.limit = limit;
        }
        
        public double[] predict() {
            if(distribution == null) {
                final PrunedMultimap<Double, Integer> copy = PrunedMultimap.asc();
                copy.putAll(neighbourMap);
                copy.setHardLimit(k);
                copy.setRandom(getRandom());
                copy.setDiscardType(PrunedMultimap.DiscardType.RANDOM);
                copy.prune();
                distribution = new double[trainData.numClasses()];
                for(Integer i : neighbourMap.values()) {
                    distribution[i]++;
                }
                ArrayUtilities.normalise(distribution);
            }
            return distribution;
        }
        
        @Override public Iterator<Double> iterator() {
            
            if(instanceExaminationReordering && k == 1) {
                final List<List<Integer>> instIndicesByClass = trainData.getInstIndicesByClass();
                
                return new Iterator<Double>() {
                    
                    private int instCount = 0;

                    @Override public boolean hasNext() {
                        if(instCount >= trainData.numInstances()) {
                            return false;
                        }
                        // get the best neighbours
                        final HashSet<Integer> nearestNeighbourClassSet = new HashSet<>(neighbourMap.get(neighbourMap.firstKey()));
                        // if there's more than one class in the best neighbours then continue - need to attempt to resolve the tie in distance for neighbours of different classes
                        if(nearestNeighbourClassSet.size() > 1) {
                            return true;
                        }
                        // by here there's only one class of nearest neighbour
                        // if there's only one class of nearest neighbour(s) then we can quit if all other classes have been exhausted
                        // i.e. if the classes to pick from are entirely contained by the homogeneous nearest neighbours then all other classes have been exhausted
//                        if(nearestNeighbourClassSet.containsAll(instsIndicesByClass.keySet())) {
//                            return false;
//                        } todo
                        return true;
                    }

                    @Override public Double next() {
                        // when handling a 1-nn, it is a waste to find the nearest neighbour exactly. Instead, we should exhaust other potentially nearer neighbours from other classes. A 1-NN uses a single neighbour and therefore only has one nearest neighbour. This neighbour has a class label, which dictates the prediction made by the 1-NN. Say our nearest neighbour is from class 2. We could have any instance from class 2 and our 1-NN output would remain the same: predict class 2. Therefore we do not need to absolute nearest neighbour when constructing a 1-NN. Instead, we want the nearest neighbour which is closer than any instance from any other class. I.e. suppose we have the following labelled rankings for nearest neighbours: 2,2,2,2,1,3,2,3,4,3,0,1,3,4, ... . As you can see, as long as we obtain any of the first four class 2 insts in that sequence as the nearest neighbour then no other class has any insts which are closer, thus the 1-NN output remains the same whether we have the very closest neighbour or the 4th closest. Thus, upon finding a nearest neighbour for 1-NN we can isolate the case of exhausting all other insts from other classes and avoid having to examine the rest of insts in class 2. The more easily split the data is, the more effect this should have.
                        // find the best neighbour's class, random tie break if necessary
                        final List<Integer> bestNeighbours = neighbourMap.get(neighbourMap.firstKey());
                        final Integer bestNeighbourClass = RandomUtils.choice(bestNeighbours, getRandom());
                        // we don't want to examine any more insts from the best class's pool as we've already got one as the nearest neighbour
                        // therefore randomly pick from the remaining classes
//                        final LinkedHashSet<Integer> pool = new LinkedHashSet<>(instsIndicesByClass.keySet());
//                        pool.remove(bestNeighbourClass);
//                        final Integer chosenClass = RandomUtils.choice(new ArrayList<>(pool), getRandom());
//                        // same an inst from the next class, i.e. give an opportunity for an inst from another class to become the nearest neighbour
//                        final List<Integer> sameClassInstIndices = instsIndicesByClass.get(chosenClass);
//                        final Integer index = RandomUtils.choiceIndex(sameClassInstIndices.size(), getRandom());
//                        // remove the inst from the pool so we don't examine it as a neighbour again
//                        final int instIndex = CollectionUtils.removeUnordered(sameClassInstIndices, index);
//                        // if there are no more insts for this class then remove the pool
//                        if(sameClassInstIndices.isEmpty()) {
//                            instsIndicesByClass.remove(chosenClass);
//                        }
                        instCount++;
                        // add the neighbour as a potential new nearest neighbour
                        return add(instIndex);
                    }
                };
            } else {
                return new Iterator<Double>() {
                    
                    private int count = 0;
                    
                    @Override public boolean hasNext() {
                        return count < trainData.numInstances();
                    }

                    @Override public Double next() {
                        return add(count++);
                    }
                };
            }
            
        }
    }

    public int getK() {
        return k;
    }

    public void setK(final int k) {
        if(k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        this.k = k;
    }

    public DistanceMeasure getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(final DistanceMeasure distanceMeasure) {
        this.distanceMeasure = Objects.requireNonNull(distanceMeasure);
    }

    public boolean isEarlyAbandonDistances() {
        return earlyAbandonDistances;
    }

    public void setEarlyAbandonDistances(final boolean earlyAbandonDistances) {
        this.earlyAbandonDistances = earlyAbandonDistances;
    }
}

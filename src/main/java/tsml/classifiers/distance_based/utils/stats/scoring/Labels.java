package tsml.classifiers.distance_based.utils.stats.scoring;

import tsml.classifiers.distance_based.utils.collections.lists.RepeatList;

import java.util.*;
import java.util.stream.Collectors;

import static utilities.ArrayUtilities.normalise;

public class Labels<A> {

    public Labels(final List<A> labels, final List<Double> weights) {
        this.labels = labels;
        this.weights = weights;
    }
    
    public Labels(final List<A> labels) {
        this(labels, new RepeatList<>(1d, labels.size()));
    }

    private final List<A> labels;
    private final List<Double> weights;
    // label set is a list of unique labels. This is required specifically for labels which are not represented, i.e. given the labels 0,0,1,1,2,2,5,5. The labels are 1,2,3 and 5. Note there is no label 4. However, there is a label 3 which is not represented at all, and appears to not be an option like label 4. This leads to incorrect entropy / scoring calculations as the number of possible partitions is incorrectly reduced. I.e. given labels 0,2 a gini score would give 0.5 for two classes: 0 and 2. If however, there is in fact 3 classes, 0, 1 and 2, the gini score will be different. Therefore this label set is designed to specify the available labels, some of which may not be present in the main labels list whatsoever. 
    private List<A> labelSet;
    private Map<A, Double> countsMap;
    private List<Double> distribution;
    private List<Double> countsList;
    
    public List<A> getLabelSet() {
        if(labelSet == null) {
            labelSet = labels.stream().distinct().collect(Collectors.toList());
        }
        return labelSet;
    }
    
    public Map<A, Double> getCountsMap() {
        if(countsMap == null) {
            countsMap = new LinkedHashMap<>(getLabelSet().size(), 1f);
            for(A i : getLabelSet()) {
                countsMap.put(i, 0d);
            }
            int i = 0;
            for(A label : labels) {
                final Double weight = weights.get(i);
                countsMap.compute(label, (key, value) -> {
                    if(value == null) {
                        throw new IllegalArgumentException(label + " not in label set " + getLabelSet());
                    } else {
                        return value + weight;
                    }
                });
                i++;
            }
            countsList = null;
        }
        return countsMap;
    }
    
    public List<Double> getDistribution() {
        if(distribution == null) {
            distribution = normalise(getCountsMap().values());
        }
        return distribution;
    }

    public void setLabelSet(final List<A> labelSet) {
        this.labelSet = labelSet;
    }

    public List<A> getLabels() {
        return labels;
    }

    public List<Double> getWeights() {
        return weights;
    }
    
    public List<Double> getCountsList() {
        if(countsList == null) {
            final Map<A, Double> countsMap = getCountsMap();
            countsList = new ArrayList<>(countsMap.values());
        }
        return countsList;
    }
}

package tsml.classifiers.distance_based.interval;

import utilities.Utilities;
import weka.core.Instances;

import java.util.List;
import java.util.function.Function;

public interface Scorer {
    double findScore(Instances parent, List<Instances> parts);

    static int[] extractSizes(Instances... parts) {
        int[] partSizes = new int[parts.length];
        for(int i = 0; i < parts.length; i++) {
            partSizes[i] = parts[i].size();
        }
        return partSizes;
    }

    static Scorer giniScore = (parent, parts) -> Utilities.giniScore(parent.size(), Utilities.convert(parts, Instances::size));
    static Scorer infoGain = (parent, parts) -> Utilities.infoGain(parent.size(), Utilities.convert(parts, Instances::size));
}

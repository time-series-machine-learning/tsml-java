package tsml.classifiers.distance_based.proximity.splitting;

import utilities.Utilities;
import weka.core.Instances;

import java.util.List;

public interface Scorer {
    double findScore(Instances parent, List<Instances> parts);

    Scorer giniScore = (parent, parts) -> Utilities.giniScore(parent.size(), Utilities.convert(parts, Instances::size));
    Scorer infoGain = (parent, parts) -> Utilities.infoGain(parent.size(), Utilities.convert(parts, Instances::size));
}

package tsml.classifiers.shapelet_based.quality;

import tsml.transformers.shapelet_tools.OrderLineObj;
import utilities.class_counts.ClassCounts;

import java.util.List;

public interface ShapeletQualityMeasureMV {

    public double calculate(List<OrderLineObj> orderline, ClassCounts classDistribution);
}

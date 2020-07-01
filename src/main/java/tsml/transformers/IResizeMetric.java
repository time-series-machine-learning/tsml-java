package tsml.transformers;

import java.util.Map;

public interface IResizeMetric {
    public int calculateResizeValue(Map<Integer, Integer> counts);
}
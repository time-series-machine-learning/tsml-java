package tsml.classifiers.distance_based.utils.collections.params;

import tsml.classifiers.distance_based.utils.collections.DefaultList;
import tsml.classifiers.distance_based.utils.collections.checks.Checks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class ParamSpace implements DefaultList<ParamMap> {

    public ParamSpace(final List<ParamMap> paramMaps) {
        addAll(paramMaps);
    }
    
    public ParamSpace() {
        
    }
    
    public ParamSpace(ParamMap... paramMap) {
        this(Arrays.asList(paramMap));
    }

    private final List<ParamMap> paramMaps = new ArrayList<>();

    public boolean add(ParamMap paramMap) {
        paramMaps.add(Objects.requireNonNull(paramMap));
        return true;
    }

    @Override public ParamMap get(final int i) {
        return paramMaps.get(i);
    }

    @Override public int size() {
        return paramMaps.size();
    }

    @Override public String toString() {
        return paramMaps.toString();
    }
    
    public ParamMap getSingle() {
        return Checks.requireSingle(paramMaps);
    }

}

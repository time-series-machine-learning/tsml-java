package utilities.params;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import tsml.classifiers.distance_based.distances.DistanceMeasureable;
import tsml.classifiers.distance_based.distances.ddtw.DDTWDistance;
import tsml.classifiers.distance_based.distances.dtw.DTW;
import tsml.classifiers.distance_based.distances.dtw.DTWDistance;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistance;
import tsml.classifiers.distance_based.distances.wddtw.WDDTWDistance;
import tsml.classifiers.distance_based.distances.wdtw.WDTW;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistance;
import utilities.ArrayUtilities;
import utilities.Utilities;
import utilities.collections.DefaultList;

/**
 * holds a mapping of parameter names to their corresponding values, where the values are stored as a ParamValues object
 * to allow for sub parameter spaces.
 */
public class ParamSpace implements DefaultList<ParamSet> { // todo don't extend list

    private Map<String, List<ParamValues>> paramsMap = new LinkedHashMap<>(); // 1-many mapping of parameter names

    public static void main(String[] args) {
        ParamSpace params = new ParamSpace();
        ParamSpace wParams = new ParamSpace();
        wParams.add(DTW.getWarpingWindowFlag(), new ParamValues(Arrays.asList(1, 2, 3, 4, 5)));
        params.add(DistanceMeasureable.getDistanceFunctionFlag(),
            new ParamValues(Arrays.asList(new DTWDistance(), new DDTWDistance()),
                Arrays.asList(wParams)));
        ParamSpace lParams = new ParamSpace();
        lParams.add(WDTW.getGFlag(), new ParamValues(Arrays.asList(1, 2, 3)));
        lParams.add(LCSSDistance.getEpsilonFlag(), new ParamValues(Arrays.asList(1, 2, 3, 4)));
        params.add(DistanceMeasureable.getDistanceFunctionFlag(),
            new ParamValues(Arrays.asList(new WDTWDistance(), new WDDTWDistance()),
                Arrays.asList(lParams)));
        int size;
        size = wParams.size();
        size = lParams.size();
        size = params.size();
        for(int i = 0; i < size; i++) {
            //            System.out.println(i);
            ParamSet param = params.get(i);
            System.out.println(param);
        }

    }

    // todo remove

    public List<Integer> getBins() {
        List<Integer> bins = new ArrayList<>();
        Iterator<Map.Entry<String, List<ParamValues>>> iterator = paramsMap.entrySet().iterator();
        for(int i = 0; i < paramsMap.size(); i++) {
            Map.Entry<String, List<ParamValues>> entry = iterator.next();
            int size = 0;
            for(ParamValues paramValues : entry.getValue()) {
                size += paramValues.size();
            }
            bins.add(size);
        }
        return bins;
    }

    public ParamSet get(int index) {
        List<Integer> indices = ArrayUtilities.fromPermutation(index, getBins());
        int i = 0;
        ParamSet param = new ParamSet();
        for(Map.Entry<String, List<ParamValues>> entry : paramsMap.entrySet()) {
            index = indices.get(i);
            List<ParamValues> paramValuesList = entry.getValue();
            for(ParamValues paramValues : paramValuesList) {
                int size = paramValues.size();
                index -= size;
                if(index < 0) {
                    Object paramValue = paramValues.get(index + size);
                    try {
                        paramValue = Utilities.deepCopy(paramValue); // must copy objects otherwise every paramset
                        // uses the same object reference!
                    } catch(Exception e) {
                        throw new IllegalStateException("cannot copy value");
                    }
                    param.add(entry.getKey(), paramValue);
                    break;
                }
            }
            if(index >= 0) {
                throw new IndexOutOfBoundsException();
            }
            i++;
        }
        return param;
    }

    public int size() {
        return ArrayUtilities.numPermutations(getBins());
    }

    public ParamSpace add(String name, ParamValues param) {
        paramsMap.computeIfAbsent(name, k -> new ArrayList<>()).add(param);
        return this;
    }

    public ParamSpace add(String name, List<?> values) {
        add(name, new ParamValues(values));
        return this;
    }

    public ParamSpace add(String name, List<?> values, List<ParamSpace> params) {
        add(name, new ParamValues(values, params));
        return this;
    }

    public ParamSpace add(String name, List<?> values, ParamSpace params) {
        add(name, values, new ArrayList<>(Collections.singletonList(params)));
        return this;
    }

    public void clear() {
        paramsMap.clear();
    }

    @Override
    public String toString() {
        return "Params" +
            //            "{" +
            //            "paramsMap=" +
            paramsMap
            //            +
            //            '}'
            ;
    }

    /**
     * holds a set of values (e.g. DTW and DDTW) and a set of corresponding params for those values (e.g. a set of
     * warping windows). I.e. many values can map to many sub spaces.
     */
    public static class ParamValues {

        private List<?> values = new ArrayList<>();
        private List<ParamSpace> paramsList = new ArrayList<>();

        public ParamValues() {
        }

        public ParamValues(List<?> values, List<ParamSpace> params) {
            setValues(values);
            setParamsList(params);
        }

        public ParamValues(List<?> values) {
            this(values, null); // no sub param space
        }

        public List<Integer> getBins() {
            List<Integer> bins = new ArrayList<>();
            for(ParamSpace paramSets : paramsList) {
                bins.add(paramSets.size());
            }
            bins.add(values.size());
            return bins;
        }

        public int size() {
            return ArrayUtilities.numPermutations(getBins());
        }

        public Object get(final int index) {
            List<Integer> indices = ArrayUtilities.fromPermutation(index, getBins());
            Object value = values.get(indices.get(indices.size() - 1));
            if(!(value instanceof ParamHandler) && !paramsList.isEmpty()) {
                throw new IllegalStateException("value not param settable");
            }
            for(int i = 0; i < paramsList.size(); i++) {
                ParamHandler paramHandler = (ParamHandler) value;
                ParamSet param = paramsList.get(i).get(indices.get(i));
                paramHandler.setParams(param);
            }
            return value;
        }

        public void addParams(ParamSpace... params) {
            this.paramsList.addAll(Arrays.asList(params));
        }

        public List<?> getValues() {
            return values;
        }

        public void setValues(List<?> values) {
            if(values == null) {
                values = new ArrayList<>();
            }
            this.values = values;
        }

        public List<ParamSpace> getParamsList() {
            return paramsList;
        }

        public void setParamsList(List<ParamSpace> paramsList) {
            if(paramsList == null) {
                paramsList = new ArrayList<>();
            }
            this.paramsList = paramsList;
        }

        @Override
        public String toString() {
            return "ParamValues{" +
                "values=" + values +
                ", params=" + paramsList +
                '}';
        }
    }


}

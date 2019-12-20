package utilities.params;

import tsml.classifiers.distance_based.distances.*;
import utilities.ArrayUtilities;
import utilities.collections.DefaultList;

import java.util.*;

/**
 * holds a mapping of parameter names to their corresponding values, where the values are stored as a ParamValues
 * object to allow for sub parameter spaces.
 */
public class ParamSpace implements DefaultList<ParamSet> {

    /**
     * holds a set of values (e.g. DTW and DDTW) and a set of corresponding params for those values (e.g. a set of
     * warping windows). I.e. many values can map to many sub spaces.
     */
    public static class ParamValues {
        private List<?> values = new ArrayList<>();
        private List<ParamSpace> paramsList = new ArrayList<>();

        public int[] getBins() {
            int[] bins = new int[paramsList.size() + 1];
            for(int i = 0; i < paramsList.size(); i++) {
                bins[i] = paramsList.get(i).size();
            }
            bins[bins.length - 1] = values.size();
            return bins;
        }

        public int size() {
            return ArrayUtilities.numPermutations(getBins());
        }

        public Object get(final int index) {
            int[] indices = ArrayUtilities.fromPermutation(index, getBins());
            Object value = values.get(indices[indices.length - 1]);
            if(!(value instanceof ParamHandler) && !paramsList.isEmpty()) {
                throw new IllegalStateException("value not param settable");
            }
            for(int i = 0; i < paramsList.size(); i++) {
                ParamHandler paramHandler = (ParamHandler) value;
                ParamSet param = paramsList.get(i).get(indices[i]);
                paramHandler.setParams(param);
            }
            return value;
        }

        public ParamValues() {}

        public ParamValues(List<?> values, List<ParamSpace> params) {
            setValues(values);
            setParamsList(params);
        }

        public ParamValues(List<?> values) {
            this(values, null); // no sub param space
        }

//        public void addValues(Object... values) {
//            this.values.addAll(Arrays.asList(values));
//        }

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

        @Override public String toString() {
            return "ParamValues{" +
                "values=" + values +
                ", params=" + paramsList +
                '}';
        }
    }

    public int[] getBins() {
        int[] bins = new int[paramsMap.size()];
        Iterator<Map.Entry<String, List<ParamValues>>> iterator = paramsMap.entrySet().iterator();
        for(int i = 0; i < bins.length; i++) {
            Map.Entry<String, List<ParamValues>> entry = iterator.next();
            int size = 0;
            for(ParamValues paramValues : entry.getValue()) {
                size += paramValues.size();
            }
            bins[i] = size;
        }
        return bins;
    }

    public ParamSet get(int index) {
        int[] indices = ArrayUtilities.fromPermutation(index, getBins());
        int i = 0;
        ParamSet param = new ParamSet();
        for(Map.Entry<String, List<ParamValues>> entry : paramsMap.entrySet()) {
            index = indices[i];
            List<ParamValues> paramValuesList = entry.getValue();
            for(ParamValues paramValues : paramValuesList) {
                int size = paramValues.size();
                index -= size;
                if(index < 0) {
                    Object paramValue = paramValues.get(index + size);
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

    private Map<String, List<ParamValues>> paramsMap = new LinkedHashMap<>(); // 1-many mapping of parameter names

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

    @Override public String toString() {
        return "Params" +
//            "{" +
//            "paramsMap=" +
            paramsMap
//            +
//            '}'
            ;
    }

    public static void main(String[] args) {
        ParamSpace params = new ParamSpace();
        ParamSpace wParams = new ParamSpace();
        wParams.add(Dtw.WARPING_WINDOW_FLAG, new ParamValues(Arrays.asList(1,2,3,4,5)));
        params.add(DistanceMeasure.DISTANCE_FUNCTION_FLAG, new ParamValues(Arrays.asList(new Dtw(), new Ddtw()),
                                                                           Arrays.asList(wParams)));
        ParamSpace lParams = new ParamSpace();
        lParams.add(Wdtw.G_FLAG, new ParamValues(Arrays.asList(1, 2, 3)));
        lParams.add(Lcss.EPSILON_FLAG, new ParamValues(Arrays.asList(1, 2, 3, 4)));
        params.add(DistanceMeasure.DISTANCE_FUNCTION_FLAG, new ParamValues(Arrays.asList(new Wdtw(), new Wddtw()),
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


//        if(!paramsList.isEmpty()) {
//            for(Object value : values) {
//                if(!(value instanceof ParamHandler)) throw new IllegalArgumentException("params cannot be applied" +
//                                                                                            " to non param " +
//                                                                                            "handling value");
//                else {
//                    for(Params params : paramsList) {
//                        if(((ParamHandler) value).listParams().contains())
//                    }
//                }
//            }
//        }
    }


}

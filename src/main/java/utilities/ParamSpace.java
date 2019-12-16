/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package utilities;

import utilities.Utilities;

import java.util.*;

public class ParamSpace {

    private Map<String, List<ParamSpace>> map = new LinkedHashMap<>();
    private List<?> values;

    public static void main(String[] args) {
        ParamSpace a = new ParamSpace();
        a.add("x", Arrays.asList(1,2,3));
        ParamSpace b = new ParamSpace();
        b.add("y", Arrays.asList(4,5,6));
        ParamSpace c = new ParamSpace();
        c.add("a", a);
        c.add("b", b);
        System.out.println(a.size());
        System.out.println(b.size());
        System.out.println(c.size());
        ParamSpace d = new ParamSpace();
        d.add("z", Arrays.asList(7,8));
        c.add("a", d);
        System.out.println(c.size());
        System.out.println(a);
        System.out.println(b);
        System.out.println(c);
        System.out.println(d);
        System.out.println(StringUtilities.prettify(a, "  "));
        int size = c.size();
        for(int i = 0; i < size; i++) {
            System.out.println(c.get(i));
        }
    }

    public List<ParamSpace> get(String name) {
        return map.get(name);
    }

    public List<?> getValues() {
        return values;
    }

    public void setValues(List<?> values) {
        this.values = values;
    }

    public void add(String name, ParamSpace space) {
        map.computeIfAbsent(name, k -> new ArrayList<>()).add(space);
    }

    public void add(String name, List<?> values) {
        ParamSpace space = new ParamSpace();
        space.setValues(values);
        add(name, space);
    }

    public void add(String name, Object[] values) {
        add(name, Arrays.asList(values));
    }

    public void add(String name, int[] values) {
        add(name, Utilities.asList(values));
    }

    public void add(String name, double[] values) {
        add(name, Utilities.asList(values));
    }

    public void add(String name, float[] values) {
        add(name, Utilities.asList(values));
    }

    public void add(String name, short[] values) {
        add(name, Utilities.asList(values));
    }

    public void add(String name, long[] values) {
        add(name, Utilities.asList(values));
    }

    public void add(String name, byte[] values) {
        add(name, Utilities.asList(values));
    }

    public void add(String name, boolean[] values) {
        add(name, Utilities.asList(values));
    }

    public void add(String name, char[] values) {
        add(name, Utilities.asList(values));
    }

    public void addAll(ParamSpace space) {
        for(Map.Entry<String, List<ParamSpace>> entry : space.map.entrySet()){
            for(ParamSpace subSpace : entry.getValue()) {

            }
        }
    }

    public void clear() {
        map.clear();
        values = null;
    }

    public List<Integer> sizes() {
        List<Integer> sizes = new ArrayList<>();
        if(values != null) {
            sizes.add(values.size());
        }
        for(Map.Entry<String, List<ParamSpace>> entry : map.entrySet() ){
            List<ParamSpace> spaces = entry.getValue();
            int size = 0;
            for(ParamSpace space : spaces) {
                size += space.size();
            }
            sizes.add(size);
        }
        return sizes;
    }

    public int size() {
        List<Integer> sizes = sizes();
        return Utilities.numPermutations(sizes);
    }

    public ParamSet get(int index) {
        List<Integer> sizes = sizes();
        List<Integer> indices = Utilities.fromPermutation(index, sizes);
        if(sizes.size() != indices.size()) throw new IllegalStateException("correctness check failed");
        ParamSet paramSet = new ParamSet();
        int i = 0;
        if(values != null) {
            int valueIndex = indices.get(i);
            i++;
            paramSet.setValue(values.get(valueIndex));
        }
        for(Map.Entry<String, List<ParamSpace>> entry : map.entrySet()) {
            String name = entry.getKey();
            List<ParamSpace> spaces = entry.getValue();
            int subIndex = indices.get(i);
            ParamSet sub = null;
            for(ParamSpace space : spaces) {
                int spaceSize = space.size();
                if(subIndex < spaceSize) {
                    sub = space.get(subIndex);
                    break;
                } else {
                    subIndex -= spaceSize;
                }
            }
            if(sizes.size() != indices.size()) throw new IllegalStateException("correctness check failed");
            paramSet.add(name, sub);
            i++;
        }
        return paramSet;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("{");
        if(values != null) {
            builder.append(values);
            if(!map.isEmpty()) {
                builder.append(", ");
            }
        }
        boolean commaBetweenNames = false;
        for(Map.Entry<String, List<ParamSpace>> entry : map.entrySet()) {
            if(commaBetweenNames) {
                builder.append(", ");
            } else {
                commaBetweenNames = true;
            }
            String name = entry.getKey();
            List<ParamSpace> spaces = entry.getValue();
            boolean commaBetweenSpaces = false;
            for(ParamSpace space : spaces) {
                if(commaBetweenSpaces) {
                    builder.append(", ");
                } else {
                    commaBetweenSpaces = true;
                }
                builder.append(name);
                if(space != null) {
                    builder.append(": ");
                    builder.append(space);
                }
            }
        }
        builder.append("}");
        return builder.toString();
    }
}

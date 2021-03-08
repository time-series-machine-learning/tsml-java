/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package experiments;

import experiments.data.DatasetLoading;
import fileIO.InFile;
import fileIO.OutFile;
import org.junit.Test;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import tsml.transformers.ROCKET;
import tsml.transformers.TrainableTransformer;
import tsml.transformers.Transformer;
import utilities.FileHandlingTools;
import weka.core.Instances;
import weka.core.Randomizable;

import java.io.File;
import java.lang.reflect.Method;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * Tests to compare test accuracies for transforms on a quick italy power
 * demand run to saved expected results
 *
 * hacked version of BasicReproductionTests
 *
 * @author Matthew Middlehurst
 */
public class BasicTransformReproductionTests {

    public static final int defaultSeed = 0;

    public static String reproductionDirectory = "src/main/java/experiments/reproductions/transforms/";

    static {
        new File(reproductionDirectory).mkdirs();
    }

    private static final String tsTransformers = "tsml.transformers.";

    public static final String[] transformerPaths = {

        tsTransformers + "Catch22",
        tsTransformers + "Differences",
        tsTransformers + "PowerSpectrum",
        tsTransformers + "ROCKET",

    };


    public static final double eps = 10e-6;
    public static boolean doubleEqual(double v1, double v2) {
        return Math.abs(v1 - v2) < eps;
    }
    public static boolean doubleArrayEquals(double[] a1, double[] a2) {
        for (int i = 0; i < a1.length; i++)
            if (!doubleEqual(a1[i], a2[i]))
                return false;
        return true;
    }

    public static class ExpectedTransformerResults {

        public String transformerName; //simple unconditioned class name
        public String fullClassName; //includes package paths for construction
        public Transformer transformer = null;
        public double[][] results;
        public String dateTime;
        public long time;

        public ExpectedTransformerResults(File resFile) throws Exception {
            transformerName = resFile.getName().split("\\.")[0]; //without any filetype extensions

            InFile in = new InFile(resFile.getAbsolutePath());
            int numLines = in.countLines();
            String[] meta = in.readLine().split(",");
            fullClassName = meta[0];
            dateTime = meta[1];
            time = Long.parseLong(meta[2]);

            String[] ln1 = in.readLine().split(",");
            results = new double[numLines-1][ln1.length];
            for (int i = 0; i < ln1.length; i++){
                results[0][i] = Double.parseDouble(ln1[i]);
            }

            for (int n = 1; n < numLines-1; n++){
                String[] ln = in.readLine().split(",");
                for (int i = 0; i < ln.length; i++){
                    results[n][i] = Double.parseDouble(ln[i]);
                }
            }
        }

        public ExpectedTransformerResults(String className) throws Exception {
            fullClassName = className;

            String[] t = fullClassName.split("\\.");
            transformerName = t[t.length-1];
        }

        public void save(String directory) throws Exception {
            directory.replace("\\", "/");
            if (!directory.endsWith("/"))
                directory+="/";

            Date date = new Date();
            SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

            dateTime = formatter.format(date);
            time = System.currentTimeMillis();

            OutFile of = new OutFile(directory + transformerName + ".csv");
            of.writeLine(fullClassName + "," + dateTime + "," + time);
            for (double[] line: results){
                StringBuilder s = new StringBuilder(Double.toString(line[0]));
                for (int i = 1; i < line.length; i++){
                    s.append(",").append(line[i]);
                }
                of.writeLine(s.toString());
            }
        }

        public boolean equal(double[][] newResults) throws Exception {
            for (int n = 0; n < newResults.length; n++){
                if (!doubleArrayEquals(newResults[n], results[n])){
                    return false;
                }
            }
            return true;
        }
    }

    public static Transformer constructTransformer(String fullTransformerName) {
        Transformer inst = null;

        try {
            Class c = Class.forName(fullTransformerName);
            inst = (Transformer) c.newInstance();

            //special cases
            if (inst instanceof ROCKET){
                ((ROCKET)inst).setNumKernels(100);
            }

            if (inst instanceof Randomizable)
                ((Randomizable)inst).setSeed(defaultSeed);
            else {
                Method[] ms = c.getMethods();
                for (Method m : ms) {
                    if (m.getName().equals("setSeed") || m.getName().equals("setRandSeed")) {
                        m.invoke(inst, defaultSeed);
                        break;
                    }
                }
            }
        } catch (Exception ex) {
            Logger.getLogger(BasicTransformReproductionTests.class.getName()).log(Level.SEVERE, null, ex);
        }

        return inst;
    }

    public static void generateMissingExpectedResults() throws Exception {
        List<String> failedTransformers = new ArrayList<>();

        List<String> existingFiles = Arrays.asList((new File(reproductionDirectory)).list());

        for (String transformerPath : transformerPaths) {
            String[] t = transformerPath.split("\\.");
            String simpleTransformerName = t[t.length-1];

            boolean exists = false;
            for (String existingFile : existingFiles) {
                if (simpleTransformerName.equals(existingFile.split("\\.")[0])) {
                    exists = true;
                    break;
                }
            }
            if (exists)
                continue;
            else {
                System.out.println("Attempting to generate missing result for " + simpleTransformerName);
            }

            if (!generateExpectedResult(transformerPath))
                failedTransformers.add(simpleTransformerName);
        }

        System.out.println("\n\n\n");
        System.out.println("Failing transformers = " + failedTransformers);
    }

    public static void generateAllExpectedResults() throws Exception {
        List<String> failedTransformers = new ArrayList<>();

        for (String transformersPath : transformerPaths) {
            String[] t = transformersPath.split("\\.");
            String simpleTransformerName = t[t.length-1];

            if (!generateExpectedResult(transformersPath))
                failedTransformers.add(simpleTransformerName);
        }

        System.out.println("\n\n\n");
        System.out.println("Failing transformers = " + failedTransformers);
    }

    public static boolean generateExpectedResult(String transformerPath) throws Exception {
        ExpectedTransformerResults expres = new ExpectedTransformerResults(transformerPath);

        boolean worked = true;

        try {
            expres.transformer = constructTransformer(transformerPath);
        } catch (Exception e) {
            System.err.println(expres.transformerName + " construction FAILED");
            System.err.println(e);
            e.printStackTrace();
            worked = false;
        }

        try {
            Instances data = DatasetLoading.sampleItalyPowerDemand(defaultSeed)[0];
            Instances t;
            if (expres.transformer instanceof TrainableTransformer){
                t = ((TrainableTransformer)expres.transformer).fitTransform(data);
            }
            else {
                t = expres.transformer.transform(data);
            }

            expres.results = new double[t.numInstances()][t.numAttributes()-1];
            for (int n = 0; n < t.numInstances(); n++) {
                for (int i = 0; i < t.numAttributes() - 1; i++) {
                    expres.results[n][i] = t.get(n).value(i);
                }
            }
        } catch (Exception e) {
            System.err.println(expres.transformerName + " evaluation on ItalyPowerDemand FAILED");
            System.err.println(e);
            e.printStackTrace();
            worked = false;
        }

        if (worked) {
            expres.save(reproductionDirectory);
            System.err.println(expres.transformerName + " evaluated and saved SUCCESFULLY");
        }

        return worked;
    }

    public static boolean confirmAllExpectedResultReproductions() throws Exception {
        System.out.println("--confirmAllExpectedResultReproductions()");

        File[] expectedResults = FileHandlingTools.listFiles(reproductionDirectory);
        if (expectedResults == null)
            throw new Exception("No expected results saved to compare to, dir="+reproductionDirectory);

        List<String> failedTransformers = new ArrayList<>();

        for (File expectedResultFile : expectedResults) {
            ExpectedTransformerResults expres = new ExpectedTransformerResults(expectedResultFile);

            Transformer transformer = constructTransformer(expres.fullClassName);
            Instances data = DatasetLoading.sampleItalyPowerDemand(defaultSeed)[0];
            Instances t;
            if (transformer instanceof TrainableTransformer){
                t = ((TrainableTransformer)transformer).fitTransform(data);
            }
            else {
                t = transformer.transform(data);
            }

            double[][] results = new double[t.numInstances()][t.numAttributes()-1];
            for (int n = 0; n < t.numInstances(); n++) {
                for (int i = 0; i < t.numAttributes() - 1; i++) {
                    results[n][i] = t.get(n).value(i);
                }
            }

            Transformer transformer2 = constructTransformer(expres.fullClassName);
            TimeSeriesInstances data2 = Converter.fromArff(DatasetLoading.sampleItalyPowerDemand(defaultSeed)[0]);
            TimeSeriesInstances t2;
            if (transformer2 instanceof TrainableTransformer){
                t2 = ((TrainableTransformer)transformer2).fitTransform(data2);
            }
            else {
                t2 = transformer2.transform(data2);
            }

            double[][] results2 = new double[t2.numInstances()][t2.getMaxLength()];
            for (int n = 0; n < t2.numInstances(); n++) {
                for (int i = 0; i < t2.getMaxLength(); i++) {
                    results2[n][i] = t2.get(n).get(0).getValue(i);
                }
            }

            if (expres.equal(results) && expres.equal(results2))
                System.out.println("\t" + expres.transformerName + " all good, parity with results created " + expres.dateTime);
            else {
                System.out.println("\t" + expres.transformerName + " was NOT recreated successfully! no parity with results created " + expres.dateTime);
                failedTransformers.add(expres.transformerName);
            }
        }

        if (failedTransformers.size() > 0) {
            System.out.println("\n\n\n");
            System.out.println("Failing classifiers = " + failedTransformers);
            return false;
        }
        return true;
    }

    @Test
    public void test() throws Exception {
        main(new String[0]);
    }

    public static void main(String[] args) throws Exception {
//        generateAllExpectedResults();
//        generateMissingExpectedResults();

        boolean transformersComplete = confirmAllExpectedResultReproductions();

        if (!transformersComplete) {
            System.out.println("Transformers simple eval recreation failed!");
            System.exit(1);
        }

        System.out.println("\n\n*********************All tests passed");
    }
}

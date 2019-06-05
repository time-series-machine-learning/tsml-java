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

package evaluation;

import ResultsProcessing.MatlabController;
import evaluation.storage.ClassifierResults;

/**
 * Class to convert results in the format as they are in the results pipeline (ClassifierResults) 
 * into the format for the roccurves.m matlab script for generating roc diagrams
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ROCDiagramMaker {

    public static String rocDiaPath = "dias_ROCCurve/";
    
    /**
     * Concatenates the predictions of classifiers made on different folds on the data
     * into one results object per classifier. 
     * 
     * @param cresults [classifier][fold]
     * @return         [classifier]
     */
    public static ClassifierResults[/*classifier*/] concatenateClassifierResults(ClassifierResults[/*classiifer*/][/*fold*/] cresults) { 
        return null;
    }
    
    public static void matlab_buildROCDiagrams(String outPath, String expName, String dsetName, ClassifierResults[][] cresults, String[] cnames) {   
        matlab_buildROCDiagrams(outPath, expName, dsetName, concatenateClassifierResults(cresults), cnames);
    }   
    
    public static void matlab_buildROCDiagrams(String outPath, String expName, String dsetName, ClassifierResults[] cresults, String[] cnames) {      
        outPath += rocDiaPath;
        
        for (PerformanceMetric metric : metrics) {
            try {
                Pair<String[], double[][]> asd = matlab_readRawFile(outPath + fileNameBuild_pws(expName, metric.name) + ".csv", dsets.length);
                ResultTable rt = new ResultTable(ResultTable.createColumns(asd.var1, dsets, asd.var2));

                int numClassiifers = rt.getColumns().size();

                MatlabController proxy = MatlabController.getInstance();
                
                for (int c1 = 0; c1 < numClassiifers-1; c1++) {
                    for (int c2 = c1+1; c2 < numClassiifers; c2++) {
                        String c1name = rt.getColumns().get(c1).getName();
                        String c2name = rt.getColumns().get(c2).getName();
                        
                        if (c1name.compareTo(c2name) > 0) {
                            String t = c1name;
                            c1name = c2name;
                            c2name = t;
                        }
                        
                        String pwFolderName = outPath + c1name + "vs" + c2name + "/";
                        (new File(pwFolderName)).mkdir();
                        
                        List<ResultColumn> pwrl = new ArrayList<>(2);
                        pwrl.add(rt.getColumn(c1name).get());
                        pwrl.add(rt.getColumn(c2name).get());
                        ResultTable pwrt = new ResultTable(pwrl);

                        proxy.eval("array = ["+ pwrt.toStringValues(false) + "];");

                        final StringBuilder concat = new StringBuilder();
                        concat.append("'");
                        concat.append(c1name.replaceAll("_", "\\\\_"));
                        concat.append("',");
                        concat.append("'");
                        concat.append(c2name.replaceAll("_", "\\\\_"));
                        concat.append("'");
                        proxy.eval("labels = {" + concat.toString() + "}");
                        
//                        System.out.println("array = ["+ pwrt.toStringValues(false) + "];");
//                        System.out.println("labels = {" + concat.toString() + "}");
//                        System.out.println("pairedscatter('" + pwFolderName + fileNameBuild_pwsInd(c1name, c2name, statName).replaceAll("\\.", "") + "',array(:,1),array(:,2),labels,'"+statName+"')");
                        
                        proxy.eval("pairedscatter('" + pwFolderName + fileNameBuild_pwsInd(c1name, c2name, metric.name).replaceAll("\\.", "") + "',array(:,1),array(:,2),labels,'"+metric.name+"','"+metric.comparisonDescriptor+"')");
                        proxy.eval("clear");
                    }
                }
            } catch (Exception io) {
                System.out.println("buildPairwiseScatterDiagrams("+outPath+") failed loading " + metric.name + " file\n" + io);
            }
        }
    
    
    
}

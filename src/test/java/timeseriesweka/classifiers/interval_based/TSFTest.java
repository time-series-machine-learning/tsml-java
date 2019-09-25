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
package timeseriesweka.classifiers.interval_based;

import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

/**
 *
 * @author xmw13bzu
 */
public class TSFTest {
    
    public TSFTest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
        System.out.println("-----Start TSF tests-----");
    }
    
    @AfterClass
    public static void tearDownClass() {
        System.out.println("-----End TSF tests-----\n");
    }
    
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }

    /**
     * Test of simple results reproduction, of class CAWPE.
     */
    @Test
    public void testReproduction() throws Exception {
        System.out.println("--testReproduction()");
        double expectedAcc = 0.967930029154519;
        System.out.println("TSF expected accuracy generated 2019_09_25");
        assertTrue(ClassifierTools.testUtils_confirmIPDReproduction(new TSF(0), expectedAcc));
    }
    
}

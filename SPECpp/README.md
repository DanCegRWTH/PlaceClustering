## SPECpp
This code is an adjusted version [eST-Miner in ProM](https://github.com/promworkbench/SPECpp) by leah-tgu. The changes adjust the eST-Miner configuration to not perform an implicit-place reduction and be able to export the given place information as a HTML file (which can then be used by our Python code). The remaining text is the original README description with an adjusted description on how to run this code.

### Original Description
This is the development repository for the bottom-up process discovery framework _SPECpp_.
An acronym for:
- **S**upervision
- **P**roposal
- **E**valuation
- **C**omposition
- **P**ost-**P**rocessing

It is intended as a standalone (with ivy dependencies) runnable framework (see package /headless, most importantly [batch CLI info](src/org/processmining/specpp/headless/batch/help.md)), however an interactive ProM plugin making use of this framework is included here (see package /prom).
This piece of software provides some structure, common implementations and extensive "supervision" as well as "inter component dependency and data management" support for interested developers who want to play around with evolutions of the original [eST-Miner](http://dx.doi.org/10.1007/978-3-030-21571-2_15) by L. Mannel et al.

The logical structure of the discovery approach is looping proposal, evaluation & composition, with post-processing at the end.
In this instance, proposal (potential candidate oracle) is specified via efficient local tree traversal.
Composition is handled by token-based replay fitness thresholding, as well as variants on "postponing" strategies that first collect a number of places (e.g. a tree level), then make slightly less greedy acceptance/rejection decisions (Delta & Uniwired Composer).
They can also make use of local information-based evaluations of the at-this-point intermediate result regarding a potential candidate. Instanced here by concurrent implicitness testing.
Finally, post-processing is a pipeline of transforming operations starting off with the final set of collected places, e.g. implicit place removal.

A big technical aspect is the "inter component dependency and data management". Components can request as well as provide arbitrarily definable dependencies, e.g. data sources & parameters, evaluation functions, observables & observers.
The at-runtime declared dependencies are resolved after constructor call and are either satisfied or not at _init()_ time.
The observables and observers are the facility by which supervision system functions. Components can publish generic performance measurement events, as well as arbitrary user defined "xy happened" events.
Concurrently running supervisors can plug into these streams of events as observers, transform it, e.g. counting, and finally log it.
Particularly for the _ProMless_ execution format, visualization components such as live updating charts are available.

#### To Run
1. Setup project java jdk (1.8)
2. Setup ivy facet in IDE (IvyIDEA-like plugin installation may be required)
   1. Mark ivysettings.xml as settings file
3. Resolve dependencies (see entry points in the next step)
4. Run entry points
   1. You can use the configurations in specpp/.run/ (IntelliJ) and the  ProM .launch files (Eclipse(r)). Using IntelliJ, it should automatically discover those entry points and display them at the top beside the run button (restart may be required, make sure to open folder `SPECpp` in IntelliJ).
   2. ProMPackageManager has to be run first to resolve ProM dependencies (this may take a while when running the first time)
   3. ProM launches this local ProM instance with the plugin in the package prom/

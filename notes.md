Stage Progress bars
    . updates but restarts throughout the cross validation through each stage.
    . we need to create like very specific checkpoints. Like after it completes all the GWR folds, thats 15% or something, then GWRF would be another 15% etc. 
    . The modeling sequence also needs to be revised. I previously had it where all five folds of GWR would run then the five folds for GWRF and so on.. thats how we need it to flow. 

Correlogram
    . Lets remove the current approach we have for creating plots in a png form.
    . lets replace it by turning the interactive plots into an artifact that you can download.
    . Have any plots use even integer steps for counts. 
    . Use drop down menu for comparing the different correlograms

Terminal Output and Stage State
    . Is there a way to implement a persitent state server for the terminal as its running in a specific stage. For example, if I start the Stage 2 - CV, then go to the results, then return to the stage 2 run, the terminal has lost the connection and if you select another stage it states that stage 2 is already running.

Move AI Assistant tab to the bottom left.
    . Create like a retro game style box come up

Build out the settings
    . Contact Us
    . About the Program and SPARC Labs (literally just me a guy. woo lol)
    . Littler easter egg game like a background snake game. a little puzzle clever spatial thang
    . Documents bundle

DAG Edge Design
    . Create easier approach for connecting edges and nodes. 
    . Allow to undo and redo
    . Allow for changing the status of the variable (confounding, treatment, etc.)

#Others#

CRS "3D" Visualization with a globe? 
    . Nothing crazy but like a nytimes style gray globe just with a simple area highlight?
    . Possibly more insight about the EPS?

Scenarios
    . Allow for true development of scenarios

Data Page
    . Collapsable Table
    . Variable Averages
    . Map viewport of variables
        - Allow to draw regions on map for specific zone statistics

No Project status correctly showing
Better project memory features

Thanks for all the help in developing this
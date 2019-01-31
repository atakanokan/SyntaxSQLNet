from flaskexample import app
from flask import request
from flask import render_template
from inference import infer_script
from utils import get_table_names, get_tables_html

# @app.route('/')
# @app.route('/index')
# def index():
#    return "Hello, World!"

@app.route('/')
@app.route('/index')
@app.route('/input')
def cesareans_input():

   return render_template("input.html")


@app.route('/output')
# @app.route('/')
# @app.route('/index')
def cesareans_output():

   # pull english question and tokenize it
   # eng_q = "What are the maximum and minimum budget of the departments?"
   eng_q = request.args.get('english_question')
   print("Question: {}".format(eng_q))

   # pull the database name for the question
   # db_name_q = 'department_management'
   db_name_q = request.args.get('database_name')
   print("Database Name: {}".format(db_name_q))

   # generate the sql query
   gen_sql = infer_script(nlq = eng_q,
                           db_name = db_name_q,
                           toy = True)
   print("Generated SQL: {}".format(gen_sql))

   # get tables from database
   tables_html, titles = get_tables_html(db_name = db_name_q)
   print("Number of Tables in DB: {}".format(len(tables_html)))


   return render_template("output.html", 
                         question = eng_q,
                         database_name = db_name_q,
                          generated_sql = gen_sql,
                          tables = tables_html,
                          titles = titles)
#!/usr/bin/python

from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from m09_model_deployment import predict
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

api = Api(
    app, 
    version='1.0', 
    title='Clasificación de generos de películas',
    description='Clasificación de generos de películas')

ns = api.namespace('predict', 
     description='Clasificador de generos de peliculas')
   
parser = api.parser()

parser.add_argument(
    'title', 
    type=str, 
    required=True, 
    help='Title', 
    location='json')

parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='Plot', 
    location='json')

parser.add_argument(
    'year', 
    type=int, 
    required=True, 
    help='Year', 
    location='json')


resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def post(self):
        args = parser.parse_args()

        return {
            "result": predict(args['title'], args['plot'], args['year'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=80)

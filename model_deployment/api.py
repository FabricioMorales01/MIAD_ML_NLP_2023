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
    title='Predicción de precios de vehículos usados API',
    description='Predicción de precios de vehículos usados API')

ns = api.namespace('predict', 
     description='Price Regression')
   
parser = api.parser()

parser.add_argument(
    'year', 
    type=int, 
    required=True, 
    help='Año del vehículo', 
    location='args')

parser.add_argument(
    'mileage', 
    type=int, 
    required=True, 
    help='Kilometraje', 
    location='args')

parser.add_argument(
    'state', 
    type=str, 
    required=True, 
    help='Estado', 
    location='args')

parser.add_argument(
    'make', 
    type=str, 
    required=True, 
    help='Marca', 
    location='args')

parser.add_argument(
    'model', 
    type=str, 
    required=True, 
    help='Modelo', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.Float,
})

@ns.route('/')
class PriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()

        return {
            "result": predict(args['year'], args['mileage'], args['state'], args['make'], args['model'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=80)

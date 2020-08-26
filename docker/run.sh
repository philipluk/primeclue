set -e
echo "Starting Primeclue frontend"
/usr/sbin/nginx
echo "Starting Primeclue backend"
echo "If this succeeds, go to http://127.0.0.1:8080"
/usr/sbin/primeclue-api

set -e
echo "Starting Primeclue frontend"
/usr/sbin/nginx
echo "Starting Primeclue backend"
/usr/sbin/primeclue-api

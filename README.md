## Create E2B template 

First create e2b.Dockerfile with this base

```
FROM e2bdev/code-interpreter:latest 

RUN pip install {Add packages that you want to use here}
```

Then run this command to build the template
``` bash
npm i @e2b/cli

npx e2b template build -c "/root/.jupyter/start-up.sh" -n TEMPLATE_NAME --cpu-count 2 --memory-mb 512 
```

Once the vm is built, you'll get an template id with which you can create sandboxes

``` python
E2B_PYTHON_AGENT_TEMPLATE_ID = "278b3hv07r3hg26jcft2"
e2b_sandbox = Sandbox(template=E2B_PYTHON_AGENT_TEMPLATE_ID)
```
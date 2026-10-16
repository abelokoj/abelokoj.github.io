---
layout: about
title: About
permalink: /
subtitle: <a href='https://www.math.fsu.edu/' target='_blank' rel='noopener'>Department of Mathematics</a>, Florida State University | 303 Love Bldg., 1017 Academic Way, TLH, FL 32306

profile:
  align: center
  image: prof_pic.jpg
  image_circular: true # crops the image to make it circular
  # more_info: >
  #   <p>3208 Love Building, 1017 Academic Way </p>
  #   <p>Tallahassee, FL 32306-4510</p>

# selected_papers: true # includes a list of papers marked as "selected={true}" (turned off for now)
selected_papers: false
social: true # includes social icons at the bottom of the page

announcements:
  enabled: true # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 6 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: true
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 10 # leave blank to include all the blog posts
---

<!-- Write your biography here. Tell the world about yourself. Link to your favorite [subreddit](https://www.reddit.com). You can put a picture in, too. The code is already in, just name your picture `prof_pic.jpg` and put it in the `img/` folder.

Put your address / P.O. box / other info right below your picture. You can also disable any of these elements by editing `profile` property of the YAML header of your `_pages/about.md`. Edit `_bibliography/papers.bib` and Jekyll will render your [publications page](/al-folio/publications/) automatically.

Link to your social media connections, too. This theme is set up to use [Font Awesome icons](https://fontawesome.com/) and [Academicons](https://jpswalsh.github.io/academicons/), like the ones below. Add your Facebook, Twitter, LinkedIn, Google Scholar, or just disable all of them. -->
 
  
My name is Abel Okojunu, and I am a PhD candidate in Applied and Computational Mathematics at Florida State University (FSU). I am in the final year of the program and will defend my dissertation in November 2026 under the supervision of Prof. Mark Sussman. My research sits at the intersection of Computational Fluid Dynamics (CFD) and numerical analysis. Particularly, I research the instability that causes the breakup of liquid jet into droplets, and then develop numerical methods that predict this directly from an existing flow simulation.   
      
My dissertation, *Global Linear Stability Analysis of Capillary-Driven Interfacial Instability: A Jacobian-Free Approach*, extends a matrix-free stability method, namely; the Navier–Stokes-based mean flow perturbation method of Ranjan et al., from single-phase flows to multi-phase flows with a sharp interface. The method uses an existing nonlinear solver, runs the flow with/without a small interface displacement/perturbation and subtracts them. Dynamic mode decomposition then extracts the instability's growth rates and modes from the difference. The solver itself, a coupled level set and continuous moment-of-fluid method (CLS-CMOF), is used without modification. For the capillary instability of a viscous liquid column, the method reproduces the exact growth rate to within half a percent across most of the unstable band.  
    
My interest in numerical methods for partial differential equations began during my undergraduate studies in Mathematics at the Federal University of Petroleum Resources, Nigeria, where my thesis was on Higher order compact finite difference schemes for BVPs. Since then, I have pursued it in two further settings: a summer intern appointment at LLNL, where I applied the Sherman–Morrison formula to accelerate an ODE solver for sustainable-aviation modeling, and several years at FSU as a graduate teaching assistant and instructor of record for Calculus I--III, and Applied Linear Algebra, and other courses. You can find out what some of my past students have to say about me here <https://www.ratemyprofessors.com/professor/2931172>  
    
My core research interests include 
- Computational Fluid Mechanics (Fluid and Gas Dynamics)
- Multiphase flows and interfacial dynamics
- Hydrodynamic and global stability analysis
- Capillary instabilities and liquid breakup
- Matrix-free and Jacobian-free methods
- Data-driven modeling and modal analysis of fluid flows

Besides these, I also have keen interest in Data Science/Analytics, SciML & AI, uncertainty quantification, Inverse problems, PINNs, and wherever Mathematical techniques, or computational methods are applied to solving humanity's challenges, pushing the boundaries of Science, and expanding our horizon. During my spare time, I sometimes write about these topics on my [blog]({{ '/blog/' | relative_url }}) and [projects]({{ '/projects/' | relative_url }}) and [publications]({{ '/publications/' | relative_url }}) section which you can find on this page.  
  
I am currently building SumatraPDF Enhanced, a lightweight pdf reader with full featured annotation tools comparably at the same level of what is obtainable with popular apps like notability, good notes, Samsung notes etc. It is highly useful tool for researchers, students, and instructors etc. If you work with pdf documents frequently, you will find SumatraPDF Enhanced to be a very useful addition/replacement for most of your current tool stack. You can find it using the link <https://github.com/abelokoj/sumatrapdf-enhanced>

<div role="note" class="site-note" style="margin-top: 2rem; padding: 0.75rem 1rem; border-left: 4px solid var(--global-theme-color); background-color: var(--global-card-bg-color); color: var(--global-text-color);">
  <strong>Note:</strong> This site is under active development. Some pages are incomplete, and content will continue to be added and revised (Including this about me page). It will continue for as long as this notification is still here. You will no longer see this notification as soon as the site development is completed.
</div>
